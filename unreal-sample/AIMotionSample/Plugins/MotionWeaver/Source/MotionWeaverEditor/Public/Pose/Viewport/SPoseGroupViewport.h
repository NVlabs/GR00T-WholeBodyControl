// Copyright ZhaoZining. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "AdvancedPreviewScene.h"
#include "Pose/Data/PoseChainTypes.h"
#include "SAssetEditorViewport.h"
#include "UObject/GCObject.h"

class UPoseGroupProxyAsset;
class UPoseComponent;
class FPoseGroupViewportClient;

/**
 * 姿态代理组的预览视口。
 *
 * 持有一个独立预览场景：每个关键帧一个真蒙皮组件，全部按
 * LocalLocation / BodyRotation 摆出来；选中帧保持实体材质，
 * 其余帧用线框幽灵显示。轨迹由 FPoseGroupViewportClient 直接画线。
 * 三轴手柄由 FPoseGroupEditMode 提供——那是引擎给资产编辑器留的位置。
 */
class SPoseGroupViewport final
	: public SAssetEditorViewport
	, public FGCObject
{
public:
	SLATE_BEGIN_ARGS(SPoseGroupViewport) {}
		SLATE_ARGUMENT(UPoseGroupProxyAsset*, GroupAsset)
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

	/** 换资产。 */
	void SetGroupAsset(UPoseGroupProxyAsset* InAsset);

	/** 切到某一帧，更新高亮并把选中组件交给手柄。 */
	void SetSelectedKey(int32 KeyIndex);

	/** 资产内容变了（增删帧、撤销、细节面板改值）后重建整个组件池。 */
	void RefreshFromAsset();

	/** 只重贴选中帧，用于拖拽结束或元素层移动。 */
	void RefreshSelectedPose();

	/** 视口里拖拽产生的改动。由编辑器接走，避免控件直接改资产。 */
	DECLARE_DELEGATE_OneParam(FOnPoseKeyDragged, const FPoseKey& /*NewKeyValue*/);
	FOnPoseKeyDragged OnKeyDragged;

	/** 拖拽结束：松手时才求解。 */
	DECLARE_DELEGATE(FOnPoseDragFinished);
	FOnPoseDragFinished OnDragFinished;

	//~ Begin FGCObject Interface
	virtual void AddReferencedObjects(FReferenceCollector& Collector) override;
	virtual FString GetReferencerName() const override { return TEXT("SPoseGroupViewport"); }
	//~ End FGCObject Interface

protected:
	//~ Begin SEditorViewport Interface
	virtual TSharedRef<FEditorViewportClient> MakeEditorViewportClient() override;
	//~ End SEditorViewport Interface

private:
	/** 让组件数量与 Keys 对齐，并把每一帧的姿态贴进去。 */
	void SyncPoseComponents();

	/** 把某一帧的位置、朝向与姿态写进对应组件。 */
	void ApplyKeyToComponent(int32 KeyIndex);

	/** 选中帧实体显示，其余帧线框幽灵显示。 */
	void ApplySelectionHighlight();

	/** 当前选中帧的组件；没有选中或越界时返回 nullptr。 */
	UPoseComponent* GetSelectedComponent() const;

	/** 创建一个已挂进预览场景的空组件。 */
	UPoseComponent* CreatePoseComponent();

	/** 取本视口的客户端，未就绪时返回 nullptr。 */
	FPoseGroupViewportClient* GetPoseViewportClient() const;

	void HandleKeyDragged(const FPoseKey& NewKeyValue);
	void HandleDragFinished();

	TSharedPtr<FAdvancedPreviewScene> PreviewScene;

	/**
	 * 每个关键帧一个组件，下标与 GroupAsset->Keys 一一对应。
	 * 没有网格的帧也保留一个隐藏组件，避免删帧后下标错位。
	 */
	TArray<TObjectPtr<UPoseComponent>> PoseComponents;

	/**
	 * 用 TObjectPtr 而不是裸指针。
	 * AddReferencedObject 的裸指针重载已弃用（C4996），
	 * TObjectPtr 版本是当前推荐写法，顺带也能被 GC 正确追踪。
	 */
	TObjectPtr<UPoseGroupProxyAsset> GroupAsset = nullptr;

	int32 SelectedKeyIndex = INDEX_NONE;
};
