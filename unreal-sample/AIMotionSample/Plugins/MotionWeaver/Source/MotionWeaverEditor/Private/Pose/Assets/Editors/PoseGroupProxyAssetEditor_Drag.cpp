// Copyright ZhaoZining. All Rights Reserved.

/**
 * 姿态代理组编辑器 · 拖动与撤销。
 *
 * 从 PoseGroupProxyAssetEditor.cpp 拆出来：那一个文件原本 379 行，
 * 把「编辑器装配」「关键帧增删改」「拖动与撤销」三件事挤在一起。
 * 拖动这部分最容易出问题（事务时机、句柄生命周期、撤销后刷新），
 * 单独成文件后改它时不用在 400 行里找。
 *
 * 实测的拖动调用链（与最初的设计预期不同，别改错地方）：
 *   Gizmo → FPoseGroupEditMode::InputDelta → ApplyDeltaToComponent
 *         → OnKeyDragged → ApplyDraggedKey（数据在这里落盘）
 *         → FinishDrag（松手求解）
 * 不走 ITypedElementWorldInterface::SetWorldTransform。
 */

#include "Pose/Assets/Editors/PoseGroupProxyAssetEditor.h"

#include "Pose/Data/PoseGroupProxyAsset.h"
#include "Pose/Elements/PoseKeyElementWorldInterface.h"
#include "Pose/Viewport/SPoseGroupViewport.h"
#include "Core/MotionWeaverLog.h"

#include "Editor.h"

#define LOCTEXT_NAMESPACE "FPoseGroupProxyAssetEditor"

void FPoseGroupProxyAssetEditor::ApplyDraggedKey(const FPoseKey& NewKeyValue)
{
	if (GroupAsset == nullptr || !GroupAsset->Keys.IsValidIndex(SelectedKeyIndex))
	{
		return;
	}

	FPoseKey& Key = GroupAsset->Keys[SelectedKeyIndex];

	// 数值没变就不重算：拖拽每帧都会回调，无差别求解会把拖动拖成卡顿。
	const bool bChanged =
		!Key.LocalLocation.Equals(NewKeyValue.LocalLocation, 0.01) ||
		!Key.BodyRotation.Equals(NewKeyValue.BodyRotation, 0.05);

	if (!bChanged)
	{
		return;
	}

	// 第一次真正改数据时开事务，整段拖动合成一个撤销点。
	//
	// 两个前提缺一不可：
	//   1. 资产要带 RF_Transactional（构造函数里补的），否则 Modify 只标脏不进栈；
	//   2. 不能调 DisableDeltaModification，那会把 Modify 挡在事务缓冲之外。
	if (!bMovementTransactionOpen)
	{
		if (GEditor != nullptr)
		{
			GEditor->BeginTransaction(LOCTEXT("DragPoseKey", "拖动关键帧"));
			bMovementTransactionOpen = true;
		}
	}

	// 开完事务再 Modify，快照才会进撤销栈。
	GroupAsset->Modify();

	Key.LocalLocation = NewKeyValue.LocalLocation;
	Key.SetBodyRotation(NewKeyValue.BodyRotation);

	// 拖拽期间不重解整条链——作者还在动，只有松手才需要最终结果。
	// 但资产要标脏，松手时统一 Solve。
	GroupAsset->MarkPackageDirty();
}

void FPoseGroupProxyAssetEditor::FinishDrag()
{
	if (GroupAsset == nullptr)
	{
		return;
	}

	if (bMovementTransactionOpen)
	{
		if (GEditor != nullptr)
		{
			GEditor->EndTransaction();
		}
		bMovementTransactionOpen = false;
	}

	GroupAsset->Solve();
	RefreshFromAsset();
}

void FPoseGroupProxyAssetEditor::SyncPreviewFromAsset()
{
	// 拖拽过程中元素层改的是资产里的 FPoseKey，
	// 预览组件不会自己动；这里把新位置推给它，画面才跟手。
	if (ViewportWidget.IsValid())
	{
		ViewportWidget->RefreshSelectedPose();
	}
}

void FPoseGroupProxyAssetEditor::BindElementMovement()
{
	UnbindElementMovement();

	// 元素层用的是静态委托（接口实例由 Registry 持有，拿不到它），
	// 所以这里用弱引用防止编辑器先销毁后回调。
	TWeakPtr<FPoseGroupProxyAssetEditor> WeakThis = SharedThis(this);

	MovementHandles.Add(UPoseKeyElementWorldInterface::OnMovementStarted().AddLambda(
		[WeakThis]()
		{
			// 这里不开事务：实际拖动不走这条链路。
			// 事务统一在 ApplyDraggedKey / FinishDrag 里开合，
			// 那是真正被调用的写入路径。
			if (const TSharedPtr<FPoseGroupProxyAssetEditor> Pinned = WeakThis.Pin())
			{
				if (Pinned->GroupAsset != nullptr)
				{
					Pinned->GroupAsset->Modify();
				}
			}
		}));

	MovementHandles.Add(UPoseKeyElementWorldInterface::OnMovementOngoing().AddLambda(
		[WeakThis]()
		{
			if (const TSharedPtr<FPoseGroupProxyAssetEditor> Pinned = WeakThis.Pin())
			{
				Pinned->SyncPreviewFromAsset();
			}
		}));

	MovementHandles.Add(UPoseKeyElementWorldInterface::OnMovementEnded().AddLambda(
		[WeakThis]()
		{
			// FinishDrag 里会收尾事务与求解，这里不重复。
			if (const TSharedPtr<FPoseGroupProxyAssetEditor> Pinned = WeakThis.Pin())
			{
				Pinned->FinishDrag();
			}
		}));
}

void FPoseGroupProxyAssetEditor::UnbindElementMovement()
{
	// 拖动中途解绑（比如直接关编辑器）时事务还开着，这里要收掉，
	// 否则会留下一个永不结束的全局事务。
	if (bMovementTransactionOpen)
	{
		if (GEditor != nullptr)
		{
			GEditor->EndTransaction();
		}
		bMovementTransactionOpen = false;
	}

	if (MovementHandles.IsValidIndex(0))
	{
		UPoseKeyElementWorldInterface::OnMovementStarted().Remove(MovementHandles[0]);
	}
	if (MovementHandles.IsValidIndex(1))
	{
		UPoseKeyElementWorldInterface::OnMovementOngoing().Remove(MovementHandles[1]);
	}
	if (MovementHandles.IsValidIndex(2))
	{
		UPoseKeyElementWorldInterface::OnMovementEnded().Remove(MovementHandles[2]);
	}
	MovementHandles.Reset();
}

void FPoseGroupProxyAssetEditor::OnPostUndoRedo()
{
	// 撤销改的是资产里的 FPoseKey，视图不会自己知道。
	// 这里重新求解一次并刷新三方，蒙皮与轨迹才会跟上回滚后的数据。
	if (GroupAsset == nullptr)
	{
		return;
	}

	GroupAsset->Solve();
	RefreshFromAsset();
}

#undef LOCTEXT_NAMESPACE
