import enum


# see more https://huggingface.co/docs/diffusers/v0.20.0/en/api/schedulers/overview

class SchedulerMethods(enum.Enum):
    # 老派采样器
    Euler = "Euler"
    EulerA = "Euler a"
    DDIM = "DDIM"
    # DPM采样器
    DPM_SDE_Karras = "DPM++ SDE Karras"
    DPM_2M_Karras = "DPM++ 2M Karras"
    DPM_2M_SDE = "DPM++ 2M SDE"
    DPM_2M_SDE_Karras = "DPM++ 2M SDE Karras"
    # 2023新采样器
    UniPCMultistep = "UniPC"


def scheduler_list():
    return ["automatic"] + [m.value for m in SchedulerMethods.__members__.values()]


def set_scheduler(pipe, method: str):
    if method == SchedulerMethods.Euler.value:
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if method == SchedulerMethods.EulerA.value:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if method == SchedulerMethods.DDIM.value:
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if method == SchedulerMethods.DPM_2M_Karras.value:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if method == SchedulerMethods.DPM_SDE_Karras:
        from diffusers import DPMSolverSinglestepScheduler
        pipe.scheduler = DPMSolverSinglestepScheduler(pipe.scheduler.config, use_karras_sigmas=True)
    if method == SchedulerMethods.DPM_2M_SDE:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
    if method == SchedulerMethods.DPM_2M_SDE_Karras:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler(pipe.scheduler.config, algorithm_type="sde-dpmsolver++",
                                                     use_karras_sigmas=True)
    if method == SchedulerMethods.UniPCMultistep.value:
        from diffusers import UniPCMultistepScheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
