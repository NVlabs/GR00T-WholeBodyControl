// Copyright ZhaoZining. All Rights Reserved.

#include "Pose/Viewport/SPoseGroupViewport.h"

#include "Pose/Data/PoseGroupProxyAsset.h"
#include "Pose/Data/PoseProxyAsset.h"
#include "Pose/Runtime/PoseComponent.h"
#include "Pose/Viewport/PoseGroupViewportClient.h"

#include "AdvancedPreviewScene.h"
#include "SEditorViewport.h"

void SPoseGroupViewport::Construct(const FArguments& InArgs)
{
	GroupAsset = InArgs._GroupAsset;

	// 预览场景自带地板与天光；整段运动会铺开，地板给个参照。
	PreviewScene = MakeShared<FAdvancedPreviewScene>(FPreviewScene::ConstructionValues());

	SelectedKeyIndex = GroupAsset != nullptr && GroupAsset->Keys.Num() > 0 ? 0 : INDEX_NONE;
	SEditorViewport::Construct(SEditorViewport::FArguments());

	SyncPoseComponents();
	SetSelectedKey(SelectedKeyIndex);
}

FPoseGroupViewportClient* SPoseGroupViewport::GetPoseViewportClient() const
{
	// GetViewportClient 在 Construct 早期可能还没有：
	// 那时控件尚未完成初始化，不能强转后调用。
	const TSharedPtr<FEditorViewportClient> Handle = GetViewportClient();
	return Handle.IsValid() ? static_cast<FPoseGroupViewportClient*>(Handle.Get()) : nullptr;
}

void SPoseGroupViewport::SetGroupAsset(UPoseGroupProxyAsset* InAsset)
{
	GroupAsset = InAsset;
	SelectedKeyIndex = InAsset != nullptr && InAsset->Keys.Num() > 0 ? 0 : INDEX_NONE;
	SyncPoseComponents();
	SetSelectedKey(SelectedKeyIndex);
}

void SPoseGroupViewport::SetSelectedKey(int32 KeyIndex)
{
	SelectedKeyIndex = KeyIndex;
	ApplySelectionHighlight();

	if (FPoseGroupViewportClient* ViewportClientPtr = GetPoseViewportClient())
	{
		ViewportClientPtr->SetSelectedKey(KeyIndex);
		ViewportClientPtr->SetPreviewComponent(GetSelectedComponent());
	}
}

void SPoseGroupViewport::RefreshFromAsset()
{
	SyncPoseComponents();
}

void SPoseGroupViewport::RefreshSelectedPose()
{
	if (GroupAsset == nullptr || !GroupAsset->Keys.IsValidIndex(SelectedKeyIndex))
	{
		return;
	}

	ApplyKeyToComponent(SelectedKeyIndex);
	ApplySelectionHighlight();

	if (FPoseGroupViewportClient* ViewportClientPtr = GetPoseViewportClient())
	{
		ViewportClientPtr->SetPreviewComponent(GetSelectedComponent());
		ViewportClientPtr->Invalidate();
	}
}

UPoseComponent* SPoseGroupViewport::CreatePoseComponent()
{
	if (!PreviewScene.IsValid())
	{
		return nullptr;
	}

	UPoseComponent* NewComponent = NewObject<UPoseComponent>(GetTransientPackage());
	if (NewComponent == nullptr)
	{
		return nullptr;
	}

	NewComponent->SetFlags(RF_Transient);
	NewComponent->SetMobility(EComponentMobility::Movable);
	NewComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	NewComponent->SetGenerateOverlapEvents(false);
	PreviewScene->AddComponent(NewComponent, FTransform::Identity);
	return NewComponent;
}

void SPoseGroupViewport::ApplyKeyToComponent(int32 KeyIndex)
{
	if (GroupAsset == nullptr
		|| !GroupAsset->Keys.IsValidIndex(KeyIndex)
		|| !PoseComponents.IsValidIndex(KeyIndex))
	{
		return;
	}

	UPoseComponent* Component = PoseComponents[KeyIndex];
	if (Component == nullptr)
	{
		return;
	}

	const FPoseKey& Key = GroupAsset->Keys[KeyIndex];

	// 位置与朝向由 key 决定；蒙皮网格本身不带根位移。
	Component->SetRelativeLocationAndRotation(Key.LocalLocation, Key.BodyRotation);

	const bool bHasMesh = Key.PoseAsset != nullptr && !Key.PoseAsset->PreviewMesh.IsNull();
	Component->SetVisibility(bHasMesh);

	if (bHasMesh)
	{
		Component->ApplyPoseAsset(Key.PoseAsset);
	}
}

void SPoseGroupViewport::SyncPoseComponents()
{
	if (!PreviewScene.IsValid())
	{
		return;
	}

	// 重建期间先把编辑模式手里的组件摘掉，避免销毁后留下悬空指针。
	if (FPoseGroupViewportClient* ViewportClientPtr = GetPoseViewportClient())
	{
		ViewportClientPtr->SetPreviewComponent(nullptr);
	}

	const int32 DesiredCount = GroupAsset != nullptr ? GroupAsset->Keys.Num() : 0;

	while (PoseComponents.Num() > DesiredCount)
	{
		TObjectPtr<UPoseComponent> Extra = PoseComponents.Pop();
		if (Extra != nullptr)
		{
			PreviewScene->RemoveComponent(Extra);
			Extra->DestroyComponent();
		}
	}

	while (PoseComponents.Num() < DesiredCount)
	{
		UPoseComponent* NewComponent = CreatePoseComponent();
		if (NewComponent == nullptr)
		{
			break;
		}
		PoseComponents.Add(NewComponent);
	}

	for (int32 KeyIndex = 0; KeyIndex < PoseComponents.Num(); ++KeyIndex)
	{
		ApplyKeyToComponent(KeyIndex);
	}

	ApplySelectionHighlight();

	if (FPoseGroupViewportClient* ViewportClientPtr = GetPoseViewportClient())
	{
		ViewportClientPtr->SetGroupAsset(GroupAsset);
		ViewportClientPtr->SetSelectedKey(SelectedKeyIndex);
		ViewportClientPtr->SetPreviewComponent(GetSelectedComponent());
	}
}

void SPoseGroupViewport::ApplySelectionHighlight()
{
	for (int32 KeyIndex = 0; KeyIndex < PoseComponents.Num(); ++KeyIndex)
	{
		UPoseComponent* Component = PoseComponents[KeyIndex];
		if (Component == nullptr)
		{
			continue;
		}

		// 选中帧保持实体并投影；其余帧用线框幽灵，避免和选中帧抢视觉焦点。
		const bool bSelected = KeyIndex == SelectedKeyIndex;
		Component->SetForceWireframe(!bSelected);
		Component->SetCastShadow(bSelected);
	}
}

UPoseComponent* SPoseGroupViewport::GetSelectedComponent() const
{
	return PoseComponents.IsValidIndex(SelectedKeyIndex)
		? PoseComponents[SelectedKeyIndex].Get()
		: nullptr;
}

TSharedRef<FEditorViewportClient> SPoseGroupViewport::MakeEditorViewportClient()
{
	TSharedRef<FPoseGroupViewportClient> NewClient =
		MakeShared<FPoseGroupViewportClient>(PreviewScene.Get(), SharedThis(this));
	NewClient->SetGroupAsset(GroupAsset);
	NewClient->SetSelectedKey(SelectedKeyIndex);

	// 预览组件就是被拖的对象：编辑模式把它交给 ApplyDeltaToComponent，
	// 所有副作用（标脏、通知移动、刷新面板）都是真发生的。
	NewClient->SetPreviewComponent(GetSelectedComponent());

	// 模式拖出来的值回到编辑器：写资产、开合事务、松手求解都在那边。
	NewClient->SetOnKeyDragged(FPoseGroupViewportClient::FOnKeyDragged::CreateSP(
		this, &SPoseGroupViewport::HandleKeyDragged));

	NewClient->SetOnDragFinished(FPoseGroupViewportClient::FOnDragFinished::CreateSP(
		this, &SPoseGroupViewport::HandleDragFinished));

	return NewClient;
}

void SPoseGroupViewport::HandleDragFinished()
{
	OnDragFinished.ExecuteIfBound();

	// 松手后从资产重新读一次：拖动期间预览用的是组件当前值，
	// 资产可能已被求解改写，这里对齐一次避免残留中间态。
	RefreshSelectedPose();
}

void SPoseGroupViewport::HandleKeyDragged(const FPoseKey& NewKeyValue)
{
	// 不回写预览组件：它的变换本来就是 ApplyDeltaToComponent 刚写好的，
	// 再拿 key 里的值覆盖一次会把它拉回上一帧的数，表现为拖动时往回荡。
	OnKeyDragged.ExecuteIfBound(NewKeyValue);
}

void SPoseGroupViewport::AddReferencedObjects(FReferenceCollector& Collector)
{
	for (TObjectPtr<UPoseComponent>& Component : PoseComponents)
	{
		Collector.AddReferencedObject(Component);
	}
	Collector.AddReferencedObject(GroupAsset);
}
