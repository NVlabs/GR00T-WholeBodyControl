// Copyright ZhaoZining. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "EdMode.h"
#include "UnrealWidgetFwd.h"

class UPoseGroupProxyAsset;
class USceneComponent;

/**
 * 姿态代理组编辑器的编辑模式。
 *
 * 为什么非要有一个 EdMode：
 * FWidget（引擎那个三轴手柄）的绘制与命中都挂在 ModeTools 的 EdMode 上——
 * UnrealWidgetRender.cpp:187 读 ModeTools->GetShowWidget()，那个函数遍历所有
 * EdMode 并取 ShouldDrawWidget() 的或；:218 又要求 UsesTransformWidget() 为真，
 * 否则直接把 CurrentAxis 置 None 后返回。资产编辑器默认没有 EdMode，
 * 于是手柄一轴都不画、鼠标也点不中。
 *
 * 这是引擎给资产编辑器留的标准入口，Persona 就是这么做
 * （PersonaModule.cpp:161 注册 FSkeletonSelectionEditMode），
 * 而不是去硬改 EngineShowFlags。
 */
class FPoseGroupEditMode final : public FEdMode
{
public:
	FPoseGroupEditMode();

	//~ Begin FEdMode Interface
	virtual bool InputDelta(FEditorViewportClient* InViewportClient, FViewport* InViewport,
		FVector& InDrag, FRotator& InRot, FVector& InScale) override;
	virtual bool StartTracking(FEditorViewportClient* InViewportClient, FViewport* InViewport) override;
	virtual bool EndTracking(FEditorViewportClient* InViewportClient, FViewport* InViewport) override;
	virtual bool AllowsViewportDragTool() const override;
	virtual bool RequiresLegacyViewportInteractions() const override;
	virtual bool AllowWidgetMove() override;
	virtual bool ShouldDrawWidget() const override;
	virtual bool UsesTransformWidget() const override;
	virtual bool UsesTransformWidget(UE::Widget::EWidgetMode CheckMode) const override;
	virtual FVector GetWidgetLocation() const override;
	virtual bool GetCustomDrawingCoordinateSystem(FMatrix& InMatrix, void* InData) override;
	virtual bool GetCustomInputCoordinateSystem(FMatrix& InMatrix, void* InData) override;
	//~ End FEdMode Interface

	/**
	 * 编辑器把当前编辑对象告诉模式。未设置时不画手柄。
	 * InPreviewComponent 是「选中帧」的组件；视口里其他关键帧的幽灵组件
	 * 不参与手柄拖拽，否则一次拖动会同时改掉多帧。
	 */
	void SetEditingContext(UPoseGroupProxyAsset* InAsset, USceneComponent* InPreviewComponent,
		int32 InSelectedKey);

	/** 拖拽写回：把组件的新变换交给外层写进资产。 */
	DECLARE_DELEGATE_TwoParams(FOnKeyDragged, const FVector& /*NewLocation*/, const FRotator& /*NewRotation*/);
	void SetOnKeyDragged(FOnKeyDragged InDelegate) { OnKeyDragged = InDelegate; }

	/** 拖拽结束：松手时才求解整条链。 */
	DECLARE_DELEGATE(FOnDragFinished);
	void SetOnDragFinished(FOnDragFinished InDelegate) { OnDragFinished = InDelegate; }

private:
	/** 当前是否可编辑：有资产、有选中帧、有预览组件。 */
	bool CanManipulate() const;

	TWeakObjectPtr<UPoseGroupProxyAsset> GroupAsset;

	/** 选中帧的预览组件。裸指针，生命周期由视口控件持有。 */
	USceneComponent* PreviewComponent = nullptr;

	int32 SelectedKeyIndex = INDEX_NONE;

	/** 是否正在拖动。InputDelta 与 EndTracking 依赖它。 */
	bool bManipulating = false;


	FOnKeyDragged OnKeyDragged;
	FOnDragFinished OnDragFinished;
};

/** 模式的注册 ID。 */
extern const FEditorModeID PoseGroupEditModeID;
