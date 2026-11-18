// Copyright ZhaoZining. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "EditorViewportClient.h"
#include "Pose/Data/PoseChainTypes.h"

class UPoseGroupProxyAsset;

/**
 * 姿态代理组编辑器的视口客户端。
 *
 * 它只负责三件事：画轨迹与选中标记、把当前编辑对象告诉编辑模式、
 * 把模式拖出来的结果转给编辑器。手柄本身的显示与命中交给
 * FPoseGroupEditMode——那是引擎给资产编辑器留的标准位置，
 * 视口这边不再对 FWidget 做任何绕过。
 */
class FPoseGroupViewportClient final : public FEditorViewportClient
{
public:
	FPoseGroupViewportClient(FPreviewScene* InPreviewScene, const TSharedRef<SEditorViewport>& InViewport);

	/** 设置要显示的资产。 */
	void SetGroupAsset(UPoseGroupProxyAsset* InAsset);

	/** 当前选中帧。同时同步给编辑模式。 */
	void SetSelectedKey(int32 InKeyIndex);

	/** 把选中帧的预览组件交给编辑模式作为拖拽对象。 */
	void SetPreviewComponent(USceneComponent* InComponent);

	/** 模式拖拽后的回调。 */
	DECLARE_DELEGATE_OneParam(FOnKeyDragged, const FPoseKey& /*NewKeyValue*/);
	void SetOnKeyDragged(FOnKeyDragged InDelegate) { OnKeyDragged = InDelegate; }

	/** 拖拽结束：松手时才求解整条链。 */
	DECLARE_DELEGATE(FOnDragFinished);
	void SetOnDragFinished(FOnDragFinished InDelegate) { OnDragFinished = InDelegate; }

	//~ Begin FEditorViewportClient Interface
	virtual void Draw(const FSceneView* View, FPrimitiveDrawInterface* PDI) override;
	virtual void Tick(float DeltaSeconds) override;
	//~ End FEditorViewportClient Interface

private:
	/** 画轨迹、空帧落点与选中圈。 */
	void DrawTrajectory(FPrimitiveDrawInterface* PDI) const;

	const FPoseKey* GetSelectedKeyPtr() const;

	/** 找到本视口对应的编辑模式并同步上下文。 */
	void SyncEditModeContext();

	TWeakObjectPtr<UPoseGroupProxyAsset> GroupAsset;

	/** 被拖拽的选中帧预览组件。裸指针，生命周期由视口控件持有。 */
	USceneComponent* PreviewComponent = nullptr;

	int32 SelectedKeyIndex = INDEX_NONE;

	bool bViewportInitialized = false;

	FOnKeyDragged OnKeyDragged;
	FOnDragFinished OnDragFinished;
};
