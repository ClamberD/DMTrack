import torch
from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator
import traceback


def get_DMTrack_build_context(config: dict):
    print('DMTrack model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_DMTrack_model(config, impl_advice),
                                lambda impl_advice: get_DMTrack_build_string(config['model']['type'], impl_advice),
                                build_sample_input_data_generator(config))


def build_DMTrack_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type == 'dinov2':
        camoe = model_config['camoe']

        if model_impl_suggestions.optimize_for_inference:
            from .DMTrack_full_finetune import DMTrackBaseline_DINOv2
            model = DMTrackBaseline_DINOv2(
                backbone,
                common_config['template_feat_size'],
                common_config['search_region_feat_size'],

                expert_r=camoe['r'],
                expert_alpha=camoe['alpha'],
                expert_dropout=camoe['dropout'],
                use_rsexpert=camoe['use_rsexpert'],
                init_method=camoe['init_method'],

                n_routed_experts=camoe['n_routed_experts'],
                n_shared_experts=camoe['n_shared_experts'],
                n_activated_experts=camoe['n_activated_experts'],
                score_func=camoe['score_func'],
                route_scale=camoe['route_scale'],
                use_load_balance=camoe['use_load_balance'],
                lfb_momentum=camoe['lfb_momentum'],
                router_norm=camoe['router_norm'],
                lfb_update_rate=camoe['lfb_update_rate'],
            )
        else:
            from .DMTrack import DMTrack_DINOv2
            model = DMTrack_DINOv2(
                backbone,
                common_config['template_feat_size'],
                common_config['search_region_feat_size'],

                expert_r=camoe['r'],
                expert_alpha=camoe['alpha'],
                expert_dropout=camoe['dropout'],
                use_rsexpert=camoe['use_rsexpert'],
                init_method=camoe['init_method'],

                n_routed_experts=camoe['n_routed_experts'],
                n_shared_experts=camoe['n_shared_experts'],
                n_activated_experts=camoe['n_activated_experts'],
                score_func=camoe['score_func'],
                route_scale=camoe['route_scale'],
                use_load_balance=camoe['use_load_balance'],
                lfb_momentum=camoe['lfb_momentum'],
                router_norm=camoe['router_norm'],
                lfb_update_rate=camoe['lfb_update_rate'],
            )
    elif model_type == 'dinov2_full_finetune':
        from .DMTrack_full_finetune import DMTrackBaseline_DINOv2
        model = DMTrackBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_DMTrack_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'DMTrack'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
