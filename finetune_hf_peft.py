import os
import json
from metaflow import FlowSpec, step, IncludeFile, Parameter, secrets, resources, secrets, retry, pypi_base, huggingface_card, kubernetes, S3
from metaflow.profilers import gpu_profile
from exceptions import GatedRepoError, GATED_HF_ORGS

@pypi_base(packages={
    'datasets': '',
    'torch': '',
    'transformers': '',
    'peft': '',
    'trl': '',
    'accelerate': '',
    'bitsandbytes': '',
    'sentencepiece': '',
    'safetensors': '',
    'requests': ''
})
class FinetuneLlama3LoRA(FlowSpec):

    script_args_file = IncludeFile(
        'script_args',
        help="JSON file containing script arguments",
        default="hf_peft_args.json"
    )

    smoke = Parameter(
        'smoke',
        type=bool,
        default=False,
        help="Flag for a smoke test"
    )

    @secrets(sources=["huggingface-token"])
    @step
    def start(self):
        from my_peft_tools import ScriptArguments
        args_dict = json.loads(self.script_args_file)
        self.script_args = ScriptArguments(**args_dict)
        if (
            self.script_args.dataset_name.split("/")[0] in GATED_HF_ORGS
            and "HF_TOKEN" not in os.environ
        ):
            raise GatedRepoError(self.script_args.dataset_name)
        self.next(self.sft)

    @gpu_profile(interval=1)
    @huggingface_card
    @secrets(sources=["huggingface-token"])
    @kubernetes(gpu=1)
    @step
    def sft(self):
        import os
        from my_peft_tools import create_model, create_trainer, save_model, get_tar_bytes
        import huggingface_hub

        huggingface_hub.login(os.environ['HF_TOKEN']) # contained in hugginface-token secret
        model, tokenizer = create_model(self.script_args)
        trainer = create_trainer(self.script_args, tokenizer, model, smoke=self.smoke, card=True)
        trainer.train()
        output_dirname, merge_output_dirname = save_model(self.script_args, trainer)
        with S3(run=self) as s3:
            s3.put('lora_adapter.tar.gz', get_tar_bytes(output_dirname))
            if merge_output_dirname:
                s3.put('lora_merged.tar.gz', get_tar_bytes(merge_output_dirname))
        self.next(self.end)

    @step
    def end(self):
        print("Training completed successfully!")


if __name__ == '__main__':
    FinetuneLlama3LoRA()
