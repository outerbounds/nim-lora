from metaflow import FlowSpec, step, secrets, kubernetes, resources, parallel, Parameter
from metaflow.profilers import gpu_profile
import subprocess
import threading
import shutil
import time
import os
from err import *

N_GPU_CARDS = 2

class LogMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.stop_event = threading.Event()
        self.latest_logs = ""
        self.latest_line = ""

    def start(self):
        self.thread = threading.Thread(target=self.monitor_logs)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def monitor_logs(self):
        while not self.stop_event.is_set():
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith(".txt")]
            assert len(log_files) == 1 or len(log_files) == 0
            if log_files:
                latest_log = max(
                    log_files,
                    key=lambda f: os.path.getmtime(os.path.join(self.log_dir, f)),
                )
                with open(os.path.join(self.log_dir, latest_log), "r") as f:
                    self.latest_logs = f.read().strip()
                    if (
                        self.latest_line == ""
                        and self.latest_logs.split("\n")[-1] != ""
                    ):
                        self.latest_line = self.latest_logs.split("\n")[-1]
                        print()
                    if self.latest_logs.split("\n")[-1] != self.latest_line:
                        print()
                        self.latest_line = self.latest_logs.split("\n")[-1]
            time.sleep(3)


@project("torchtune-multigpu")
class FinetuneLlama3LoRA(FlowSpec):

    # https://github.com/pytorch/torchtune/blob/15c918d65d79e03abcbd0c5e94d2b116bd368412/torchtune/_cli/download.py#L57
    hf_repo_id = Parameter("repo-id", default="meta-llama/Meta-Llama-3-8B-Instruct")

    # torchtune comes with these:
    dataset = Parameter("data", help="Which dataset to use?", default="alpaca_dataset")
    # base types include: PackedDataset, ConcatDataset, TextCompletionDataset, ChatDataset, InstructionDataset, PreferenceDataset
    # example: alpaca_dataset is an InstructionDataset, slimorca_dataset is a ChatDataset, ...

    local_checkpoint_in_path = Parameter(
        "in-chkpt",
        help="Where to store the checkpoint locally?",
        default="/tmp/checkpoint-in",
    )
    local_checkpoint_out_path = Parameter(
        "out-chkpt",
        help="Where to store the checkpoint locally?",
        default="/tmp/checkpoint-out",
    )

    # https://pytorch.org/torchtune/stable/deep_dives/recipe_deepdive.html#recipe-deepdive
    workflow = Parameter(
        "workflow",
        help="What type of workflow / recipe?",
        default="lora_finetune_distributed",
    )
    # https://pytorch.org/torchtune/stable/deep_dives/configs.html#config-tutorial-label
    config = Parameter("config", help="Torchtune config", default="llama3/8B_lora")
    # To see all combinations of recipe and config, run `tune ls`.

    @secrets(sources=["huggingface-token"])
    @resources(memory=16000)
    @step
    def start(self):
        """
        Check that the necessary credentials are accessible in @secrets and config values are sensible.
        """
        # Check the dataset is legit.
        from torchtune import datasets as tt_datasets

        assert (
            self.dataset in tt_datasets.__all__
        ), f"Choose a dataset from this list: {tt_datasets.__all__}"

        # Huggingface repos routes are like 'org/model'.
        if (
            self.hf_repo_id.split("/")[0] in GATED_HF_ORGS
            and "HF_TOKEN" not in os.environ
        ):
            raise GatedRepoError(self.hf_repo_id)
        self.next(self.tune, num_parallel=1)

    @secrets(sources=["huggingface-token"])
    @gpu_profile(interval=1)
    @parallel
    # @resources(gpu=N_GPU_CARDS, memory=60000)
    @step
    def tune(self):
        """
        Download the data and run a workflow
        """
        import yaml

        # Get the model.
        # https://github.com/pytorch/torchtune/blob/15c918d65d79e03abcbd0c5e94d2b116bd368412/torchtune/_cli/download.py#L126C13-L132C14
        self.download_cmd = [
            "tune",
            "download",
            self.hf_repo_id,
            "--hf-token",
            os.environ.get("HF_TOKEN"),
            "--output-dir",
            self.local_checkpoint_in_path,
        ]
        is_success, stderr = self._exec(self.download_cmd)
        if not is_success:
            raise Exception(stderr)

        cp_cmd = ["tune", "cp", self.config, "tmp_config.yaml"]
        is_success, stderr = self._exec(cp_cmd)
        with open("tmp_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg["metric_logger"]["log_dir"] = os.path.join(os.getcwd(), "logs")
        with open("tmp_config.yaml", "w") as f:
            yaml.dump(cfg, f)
        if len(os.listdir(cfg["metric_logger"]["log_dir"])) >= 1:
            shutil.rmtree(cfg["metric_logger"]["log_dir"])
            os.makedirs(cfg["metric_logger"]["log_dir"])

        log_monitor = LogMonitor(cfg["metric_logger"]["log_dir"])
        log_monitor.start()
        self.run_cmd = [
            "tune",
            "run",
            "--nproc_per_node",
            str(N_GPU_CARDS),
            "--master_port=25678",
            self.workflow,
            "--config",
            "tmp_config.yaml",
            f"dataset=torchtune.datasets.{self.dataset}",
            f"tokenizer.path={self.local_checkpoint_in_path}/original/tokenizer.model",
            f"checkpointer.checkpoint_dir={self.local_checkpoint_in_path}/original",
            f"checkpointer.output_dir={self.local_checkpoint_out_path}/new",
            "batch_size=2",
        ]
        success, stderr = self._exec(self.run_cmd)
        log_monitor.stop()
        self.latest_logs = log_monitor.latest_logs
        if not success:
            raise Exception(stderr)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

    def _exec(self, cmd):
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            while proc.poll() is None:
                stdout = proc.stdout.read1()
                try:
                    text = stdout.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                print(text, end="", flush=True)
            if proc.returncode != 0:
                if proc.stderr is not None:
                    return False, proc.stderr.read().decode("utf-8")
                else:
                    return False, None
            return True, None


if __name__ == "__main__":
    FinetuneLlama3LoRA()
