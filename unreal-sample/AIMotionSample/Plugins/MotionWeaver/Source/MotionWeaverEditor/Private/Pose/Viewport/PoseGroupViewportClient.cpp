// Copyright ZhaoZining. All Rights Reserved.

#include "Pose/Viewport/PoseGroupViewportClient.h"

#include "Pose/PoseGroupEditMode.h"
#include "Pose/Data/PoseGroupProxyAsset.h"
#include "Pose/Data/PoseProxyAsset.h"

#include "EditorModeManager.h"
#include "EditorViewportClient.h"
#include "PrimitiveDrawingUtils.h"
#include "SEditorViewport.h"
#include "SceneManagement.h"

namespace
{
	const FLinearColor TrajectoryColor(0.16f, 0.63f, 0.86f);
	const FLinearColor GhostKeyColor(1.0f, 0.67f, 0.18f);
	const FLinearColor SelectedRingColor(0.31f, 1.0f, 0.55f);

	constexpr float TrajectoryThickness = 2.0f;
}

FPoseGroupViewportClient::FPoseGroupViewportClient(
	FPreviewScene* InPreviewScene, const TSharedRef<SEditorViewport>& InViewport)
	: FEditorViewportClient(nullptr, InPreviewScene, InViewport)
{
	// 构造里只做纯数据初始化。视角与手柄模式放到首次 Tick：
	// SetViewLocation 走 FSlateApplication，构造期 GUI 未必就绪；
	// ModeTools 也在父类构造函数靠后位置才创建（EditorViewportClient.cpp:590）。
	SetRealtime(true);
	DrawHelper.bDrawGrid = true;
	EngineShowFlags.SetGrid(true);
}

void FPoseGroupViewportClient::SetGroupAsset(UPoseGroupProxyAsset* InAsset)
{
	GroupAsset = InAsset;
	SyncEditModeContext();
	Invalidate();
}

void FPoseGroupViewportClient::SetSelectedKey(int32 InKeyIndex)
{
	SelectedKeyIndex = InKeyIndex;
	SyncEditModeContext();
	Invalidate();
}

void FPoseGroupViewportClient::SetPreviewComponent(USceneComponent* InComponent)
{
	PreviewComponent = InComponent;
	SyncEditModeContext();
}

const FPoseKey* FPoseGroupViewportClient::GetSelectedKeyPtr() const
{
	const UPoseGroupProxyAsset* Asset = GroupAsset.Get();
	if (Asset == nullptr || !Asset->Keys.IsValidIndex(SelectedKeyIndex))
	{
		return nullptr;
	}
	return &Asset->Keys[SelectedKeyIndex];
}

void FPoseGroupViewportClient::SyncEditModeContext()
{
	if (!ModeTools.IsValid())
	{
		return;
	}

	if (FPoseGroupEditMode* EditMode = ModeTools->GetActiveModeTyped<FPoseGroupEditMode>(PoseGroupEditModeID))
	{
		EditMode->SetEditingContext(GroupAsset.Get(), PreviewComponent, SelectedKeyIndex);
		EditMode->SetOnDragFinished(FPoseGroupEditMode::FOnDragFinished::CreateLambda(
			[this]()
			{
				OnDragFinished.ExecuteIfBound();
			}));

		EditMode->SetOnKeyDragged(FPoseGroupEditMode::FOnKeyDragged::CreateLambda(
			[this](const FVector& NewLocation, const FRotator& NewRotation)
			{
				if (const FPoseKey* Selected = GetSelectedKeyPtr())
				{
					FPoseKey NewKey = *Selected;
					NewKey.LocalLocation = NewLocation;
					NewKey.BodyRotation = NewRotation;
					NewKey.HeadingDegrees = NewRotation.Yaw;
					OnKeyDragged.ExecuteIfBound(NewKey);
				}
			}));
	}
}

void FPoseGroupViewportClient::Tick(float DeltaSeconds)
{
	FEditorViewportClient::Tick(DeltaSeconds);

	if (!bViewportInitialized)
	{
		bViewportInitialized = true;

		SetViewLocation(FVector(-420.0, -320.0, 240.0));
		SetViewRotation(FRotator(-18.0, 30.0, 0.0));

		// 激活编辑模式：不激活时 ModeTools 里没有 EdMode，
		// ShouldDrawWidget 无从问起，手柄依旧不画。
		if (ModeTools.IsValid())
		{
			ModeTools->ActivateMode(PoseGroupEditModeID);
			ModeTools->SetWidgetMode(UE::Widget::WM_Translate);
		}

		SyncEditModeContext();
	}

	// 预览世界的 Tick：少了它组件注册与渲染状态更新滞后，
	// 表现为「拖完要动一下鼠标才刷新」。
	if (PreviewScene != nullptr)
	{
		if (UWorld* PreviewWorld = PreviewScene->GetWorld())
		{
			PreviewWorld->Tick(IsRealtime() ? LEVELTICK_ViewportsOnly : LEVELTICK_TimeOnly, DeltaSeconds);
		}
	}
}

void FPoseGroupViewportClient::Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI)
{
	FEditorViewportClient::Draw(View, PDI);

	if (PDI == nullptr || !GroupAsset.IsValid())
	{
		return;
	}

	// 幽灵骨架现在是预览场景里的真组件，不在这里画。
	// 这里只补轨迹、空帧落点与选中圈，避免和引擎手柄抢视觉焦点。
	DrawTrajectory(PDI);
}

void FPoseGroupViewportClient::DrawTrajectory(FPrimitiveDrawInterface* PDI) const
{
	const UPoseGroupProxyAsset* Asset = GroupAsset.Get();
	if (Asset == nullptr)
	{
		return;
	}

	const TArray<FChainTrajectorySample>& Samples = Asset->GetSamples();

	for (int32 Index = 1; Index < Samples.Num(); ++Index)
	{
		PDI->DrawLine(
			Samples[Index - 1].LocalLocation,
			Samples[Index].LocalLocation,
			TrajectoryColor, SDPG_Foreground, TrajectoryThickness);
	}

	// 没有蒙皮的帧没有实体可看，至少把落点标出来。
	for (int32 KeyIndex = 0; KeyIndex < Asset->Keys.Num(); ++KeyIndex)
	{
		const FPoseKey& Key = Asset->Keys[KeyIndex];
		const bool bHasMesh = Key.PoseAsset != nullptr && !Key.PoseAsset->PreviewMesh.IsNull();
		if (bHasMesh)
		{
			continue;
		}

		const bool bSelected = KeyIndex == SelectedKeyIndex;
		const FVector MarkerLocation(
			Key.LocalLocation.X,
			Key.LocalLocation.Y,
			FMath::Max(Key.LocalLocation.Z + 2.0, 2.0));
		PDI->DrawPoint(
			MarkerLocation,
			bSelected ? SelectedRingColor : GhostKeyColor,
			bSelected ? 10.0f : 7.0f,
			SDPG_Foreground);
	}

	// 选中帧的地面圆环：线框幽灵与实体蒙皮都适用，
	// 在角色重叠时也能一眼看出当前在编辑哪一帧。
	if (Asset->Keys.IsValidIndex(SelectedKeyIndex))
	{
		const FPoseKey& Key = Asset->Keys[SelectedKeyIndex];
		const FRotator YawOnly(0.0, Key.BodyRotation.Yaw, 0.0);
		const FVector XAxis = YawOnly.RotateVector(FVector::ForwardVector);
		const FVector YAxis = YawOnly.RotateVector(FVector::RightVector);
		const FVector RingCenter(
			Key.LocalLocation.X,
			Key.LocalLocation.Y,
			FMath::Max(Key.LocalLocation.Z + 2.0, 2.0));

		DrawCircle(
			PDI,
			RingCenter,
			XAxis, YAxis,
			SelectedRingColor,
			30.0f, 32, SDPG_Foreground, 2.0f);
	}
}
