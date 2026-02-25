import os.path
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    def __init__(self, net, train_dataset, test_dataset, args, conf, device=None):
        self.args = args
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            # batch_size=args.batch_size,
            batch_size=1,
            shuffle=True,
            # num_workers=1,    # lead to very slow training
            pin_memory=False,
        )
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            # batch_size=min(args.batch_size, 16),
            batch_size=1,
            shuffle=True,
            # num_workers=1,    # lead to very slow training
            pin_memory=False,
        )

        self.num_total_batches = len(self.train_dataset)
        self.exp_name = args.name
        self.save_interval = conf.get_int("save_interval")
        self.print_interval = conf.get_int("print_interval")
        self.vis_interval = conf.get_int("vis_interval")
        self.eval_interval = conf.get_int("eval_interval")
        self.test_interval = self.eval_interval
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = conf.get_int("num_epochs", 8)
        self.accu_grad = conf.get_int("accu_grad", 1)
        self.summary_path = os.path.join(args.logs_path, args.name)
        self.writer = SummaryWriter(self.summary_path)

        # Initialize wandb if enabled
        self.use_wandb = hasattr(args, "use_wandb") and args.use_wandb
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not installed, disabling wandb logging")
                self.use_wandb = False
            else:
                wandb_config = {
                    "experiment_name": args.name,
                    "stage": args.stage,
                    "learning_rate": args.lr,
                    "datadir": args.datadir,
                    "ray_batch_size": args.ray_batch_size,
                }
                wandb.init(
                    project=getattr(args, "wandb_project", "NeRRF"),
                    entity=getattr(args, "wandb_entity", None),
                    name=f"{args.name}_stage{args.stage}",
                    config=wandb_config,
                    resume="allow",
                    dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "wandb"),
                )

        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        os.makedirs(self.summary_path, exist_ok=True)

        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optim, gamma=args.gamma
            )
        else:
            self.lr_scheduler = None

        # Load weights
        self.managed_weight_saving = hasattr(net, "load_weights")
        if self.managed_weight_saving:
            net.load_weights(self.args)
        self.iter_state_path = "%s/%s/%d/_iter" % (
            self.args.checkpoints_path,
            self.args.name,
            self.args.stage,
        )
        self.optim_state_path = "%s/%s/%d/_optim" % (
            self.args.checkpoints_path,
            self.args.name,
            self.args.stage,
        )
        self.lrsched_state_path = "%s/%s/%d/_lrsched" % (
            self.args.checkpoints_path,
            self.args.name,
            self.args.stage,
        )
        self.default_net_state_path = "%s/%s/%d/net" % (
            self.args.checkpoints_path,
            self.args.name,
            self.args.stage,
        )
        self.start_iter_id = 0
        if args.resume:
            if os.path.exists(self.default_net_state_path):
                net.load_state_dict(
                    torch.load(self.default_net_state_path, map_location=device)
                )

        self.visual_path = os.path.join(self.args.visual_path, self.args.name)
        self.conf = conf

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self, global_step, save_mesh=True):
        """
        Ran at each save step for saving extra state.
        save_mesh: if False, skip saving mesh to learned_geo (only save other state).
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def optimize_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def test_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None, None, None

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (
                " ".join(k + ": {:.6f}".format(losses[k]) for k in losses)
            )

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)
        step_id = self.start_iter_id
        total_iters = self.num_epochs * self.num_total_batches * self.num_epoch_repeats
        progress = tqdm.tqdm(total=total_iters, initial=step_id,
                             desc="Training", unit="it")
        for epoch in range(self.num_epochs):
            self.writer.add_scalar(
                "lr", self.optim.param_groups[0]["lr"], global_step=step_id
            )
            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_data_loader:
                    losses = self.train_step(data, global_step=step_id)
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        lr = self.optim.param_groups[0]["lr"]
                        progress.set_description(f"E{epoch}")
                        progress.set_postfix_str(f"B {batch} {loss_str}  lr {lr:.2e}")
                        # Log losses to wandb
                        if self.use_wandb:
                            wandb_log = {"epoch": epoch, "batch": batch, "lr": self.optim.param_groups[0]["lr"]}
                            for k, v in losses.items():
                                wandb_log[f"loss/{k}"] = v
                            wandb.log(wandb_log, step=step_id)

                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("saving")
                        if self.managed_weight_saving and self.args.stage != 1:
                            self.net.save_weights(self.args)
                        elif self.args.stage == 2:
                            torch.save(
                                self.net.state_dict(), self.default_net_state_path
                            )
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(
                                self.lr_scheduler.state_dict(),
                                self.lrsched_state_path,
                            )
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        if self.args.stage == 1:
                            self.extra_save_state(global_step=step_id, save_mesh=False)
                    ############################

                    if (
                        batch % self.vis_interval == self.vis_interval - 1
                        and self.args.stage != 1
                    ):
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals, *_ = self.vis_step(
                                test_data, global_step=step_id
                            )
                        if vis_vals is not None:
                            self.writer.add_scalars(
                                "vis", vis_vals, global_step=step_id
                            )
                        self.net.train()
                        if vis is not None:
                            import imageio

                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path,
                                    "{:01}_{:04}_{:04}_vis.png".format(
                                        self.args.stage, epoch, batch
                                    ),
                                ),
                                vis_u8,
                            )

                    if (
                        batch == self.num_total_batches - 1
                        or batch % self.accu_grad == self.accu_grad - 1
                    ):
                        # Sanitize NaN/inf gradients to prevent corruption
                        # of float16 hash grid parameters
                        for p in self.net.parameters():
                            if p.grad is not None:
                                torch.nan_to_num(
                                    p.grad, nan=0.0, posinf=1e3, neginf=-1e3,
                                    out=p.grad,
                                )
                        self.optim.step()
                        self.optim.zero_grad()

                    self.post_batch(epoch, batch)
                    step_id += 1
                    batch += 1
                    progress.update(1)
            # End of epoch: evaluate one image from val set and log to wandb
            if self.use_wandb:
                print(f"\nEpoch {epoch} finished, running validation for wandb...")
                test_data = next(iter(self.test_data_loader))
                self.net.eval()
                with torch.no_grad():
                    vis, vis_vals, gt_img, novel_view = self.vis_step(test_data, global_step=step_id)
                self.net.train()
                
                wandb_log = {"epoch": epoch}
                if vis_vals is not None:
                    for k, v in vis_vals.items():
                        wandb_log[f"val/{k}"] = v
                if gt_img is not None and novel_view is not None:
                    # Ground truth on left, novel view on right, no margin
                    side_by_side = np.concatenate([gt_img, novel_view], axis=1)
                    side_by_side_u8 = (np.clip(side_by_side, 0, 1) * 255).astype(np.uint8)
                    wandb_log["val/image"] = wandb.Image(side_by_side_u8, caption=f"Epoch {epoch}")
                elif vis is not None:
                    vis_u8 = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
                    wandb_log["val/image"] = wandb.Image(vis_u8, caption=f"Epoch {epoch}")
                wandb.log(wandb_log, step=step_id)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

        if self.args.stage == 1:
            self.extra_save_state(global_step=step_id, save_mesh=True)
