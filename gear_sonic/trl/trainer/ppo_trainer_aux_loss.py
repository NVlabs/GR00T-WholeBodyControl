import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer


class TRLAuxLossPPOTrainer(TRLPPOTrainer):
    """PPO trainer extended with per-step auxiliary losses.

    Subclasses :class:`TRLPPOTrainer` to add support for auxiliary losses
    that the policy's forward pass returns alongside the standard PPO
    objective.  Typical use-case is SONIC / universal token training where
    the ``UniversalTokenModule`` emits reconstruction and latent-alignment
    losses together with the action mean.

    The total loss is::

        loss = ppo_loss + aux_loss_scale * sum(coef_i * aux_loss_i)

    Auxiliary losses and their per-loss coefficients are expected in the
    ``policy_results`` dict under the keys ``"aux_losses"`` and
    ``"aux_loss_coef"`` respectively.

    Config keys (read from ``self.config``):

    * ``aux_loss_scale`` (float, default 1.0) – global scale applied to the
      weighted sum of auxiliary losses.
    * ``compute_aux_loss`` (bool, default ``True``) – disable to skip all
      auxiliary loss computation (useful for ablations).
    """

    _tag_names = ["trl", "aux_loss_ppo"]

    def _init_config(self):
        """Extend base config initialisation with auxiliary loss settings.

        Reads ``aux_loss_scale`` and ``compute_aux_loss`` from
        ``self.config`` and stores them as instance attributes.
        """
        super()._init_config()

        # Auxiliary loss configuration
        # aux_loss_scale: overall scalar to scale the total auxiliary loss
        self.aux_loss_scale = self.config.get("aux_loss_scale", 1.0)
        self.compute_aux_loss = self.config.get("compute_aux_loss", True)

    def _register_stats_buffer(self):
        """Allocate per-step statistics tensors for auxiliary losses.

        Calls the parent method first to register base PPO stats, then
        allocates the following additional buffers when
        ``self.compute_aux_loss`` is ``True``:

        * ``self.aux_loss_stats`` – empty dict; individual loss tensors are
          added lazily on first occurrence (see :meth:`_update_stats_buffer`).
        * ``self.total_aux_loss_unscaled_stats`` – shape
          ``(num_ppo_epochs, num_mini_batches, num_micro_batches)``, stores
          the coefficient-weighted sum before the global scale.
        * ``self.total_aux_loss_stats`` – same shape, stores the fully scaled
          total auxiliary loss.
        """
        super()._register_stats_buffer()

        if self.compute_aux_loss:
            args = self.args
            device = self.accelerator.device

            stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
            # Store stats as dictionaries to support multiple auxiliary losses
            self.aux_loss_stats = {}
            self.total_aux_loss_unscaled_stats = torch.zeros(stats_shape, device=device)
            self.total_aux_loss_stats = torch.zeros(stats_shape, device=device)

    def _extract_aux_losses_from_forward_results(self, forward_results):
        """Pull auxiliary losses and their coefficients from the policy output dict.

        Args:
            forward_results: Dict returned by the policy's forward pass.  The
                following optional keys are consumed:

                * ``"aux_losses"`` – dict mapping loss name to scalar tensor.
                * ``"aux_loss_coef"`` – dict mapping loss name to float
                  coefficient; defaults to ``None`` when absent.

        Returns:
            Tuple of ``(aux_losses_dict, aux_loss_coef)`` where
            ``aux_losses_dict`` maps loss name to tensor and
            ``aux_loss_coef`` maps loss name to float (or ``None`` when the
            key was not present in ``forward_results``).
        """
        aux_losses_dict = {}
        aux_loss_coef = None

        if "aux_losses" in forward_results:
            aux_losses_dict = forward_results["aux_losses"]

        # Extract coefficients if provided in forward_results
        if "aux_loss_coef" in forward_results:
            aux_loss_coef = forward_results["aux_loss_coef"]

        return aux_losses_dict, aux_loss_coef

    def _compute_aux_loss(self, policy_results, mb_rollout_data):
        """Compute the weighted auxiliary loss for one mini-batch.

        Extracts individual auxiliary losses and their coefficients from
        ``policy_results``, computes the per-loss weighted sum, and applies
        the global ``aux_loss_scale``.

        Args:
            policy_results: Dict from the policy forward pass (the value
                stored at ``forward_results["policy_results"]``).  Expected
                to contain ``"aux_losses"`` and optionally ``"aux_loss_coef"``.
            mb_rollout_data: Mini-batch rollout dict (not used directly but
                available for subclass overrides).

        Returns:
            Dict with keys:

            * ``"aux_losses_dict"`` – raw loss tensors keyed by name.
            * ``"aux_loss_coef"`` – per-loss coefficient dict.
            * ``"total_aux_loss_unscaled"`` – weighted sum before
              ``aux_loss_scale``, shape ``()``.
            * ``"total_aux_loss"`` – final scaled auxiliary loss,
              shape ``()``.
        """
        device = self.accelerator.device

        # Extract auxiliary losses and coefficients from forward results
        aux_losses_dict, aux_loss_coef = self._extract_aux_losses_from_forward_results(
            policy_results
        )

        if not aux_losses_dict:
            # No auxiliary losses found
            return {
                "aux_losses_dict": {},
                "aux_loss_coef": {},
                "total_aux_loss_unscaled": torch.tensor(0.0, device=device),
                "total_aux_loss": torch.tensor(0.0, device=device),
            }

        # Compute weighted sum of auxiliary losses
        total_aux_loss_unscaled = torch.tensor(0.0, device=device)
        for loss_name, loss_value in aux_losses_dict.items():
            coef = aux_loss_coef.get(loss_name, 0.0)
            total_aux_loss_unscaled += coef * loss_value

        # Apply overall scale
        total_aux_loss = total_aux_loss_unscaled * self.aux_loss_scale

        return {
            "aux_losses_dict": aux_losses_dict,
            "aux_loss_coef": aux_loss_coef,
            "total_aux_loss_unscaled": total_aux_loss_unscaled,
            "total_aux_loss": total_aux_loss,
        }

    def _compute_loss(self, forward_results, mb_rollout_data):
        """Compute total training loss as PPO loss plus auxiliary loss.

        Calls the parent ``_compute_loss`` for the standard PPO objective
        (and optionally imitation BC loss), then adds the auxiliary loss
        computed from ``forward_results["policy_results"]`` when
        ``self.compute_aux_loss`` is ``True``.

        Args:
            forward_results: Dict returned by the full forward pass.  Must
                contain ``"policy_results"`` (the policy module's output dict)
                plus whatever the parent method expects.
            mb_rollout_data: Mini-batch of rollout transitions used to
                compute advantages, returns, and old log-probs.

        Returns:
            Dict with at minimum:

            * ``"loss"`` – scalar total loss tensor (PPO + aux).
            * ``"aux_loss_dict"`` – result dict from
              :meth:`_compute_aux_loss` (only present when
              ``compute_aux_loss`` is ``True``).
            * All keys returned by the parent ``_compute_loss``.
        """
        # Compute PPO loss (includes ppo_loss and optionally imgaug_bc_loss)
        loss_dict = super()._compute_loss(forward_results, mb_rollout_data)

        # Compute and add auxiliary loss if enabled
        if self.compute_aux_loss:
            aux_loss_result = self._compute_aux_loss(
                forward_results["policy_results"], mb_rollout_data
            )

            loss_dict["loss"] += aux_loss_result["total_aux_loss"]

            # Add auxiliary loss dict to return dict
            loss_dict["aux_loss_dict"] = aux_loss_result

        return loss_dict

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,
        mb_rollout_data,
    ):
        """Record per-step loss values into pre-allocated statistics buffers.

        Delegates to the parent method for base PPO stats, then writes
        individual auxiliary loss values and aggregate totals into
        ``self.aux_loss_stats``, ``self.total_aux_loss_unscaled_stats``, and
        ``self.total_aux_loss_stats``.  Individual loss buffers are lazily
        initialised on first encounter.

        Args:
            ppo_epoch_idx: Index of the current PPO epoch (0-based).
            minibatch_idx: Index of the current mini-batch within the epoch.
            microbatch_idx: Index of the current micro-batch within the
                mini-batch (used for gradient accumulation).
            loss_dict: Output of :meth:`_compute_loss` for this step.
                Expected to contain ``"aux_loss_dict"`` when
                ``compute_aux_loss`` is ``True``.
            forward_results: Full forward-pass result dict (passed through to
                the parent method).
            mb_rollout_data: Mini-batch rollout dict (passed through to the
                parent method).
        """
        # Update PPO stats
        super()._update_stats_buffer(
            ppo_epoch_idx,
            minibatch_idx,
            microbatch_idx,
            loss_dict,
            forward_results,
            mb_rollout_data,
        )

        # Update auxiliary loss stats if enabled
        if self.compute_aux_loss and "aux_loss_dict" in loss_dict:
            aux_loss_result = loss_dict["aux_loss_dict"]
            aux_losses_dict = aux_loss_result["aux_losses_dict"]
            total_aux_loss_unscaled = aux_loss_result["total_aux_loss_unscaled"]
            total_aux_loss = aux_loss_result["total_aux_loss"]

            # Update stats for each individual auxiliary loss
            for loss_name, loss_value in aux_losses_dict.items():
                if loss_name not in self.aux_loss_stats:
                    # Lazily initialize stats buffer for this loss
                    args = self.args
                    device = self.accelerator.device
                    stats_shape = (
                        args.num_ppo_epochs,
                        args.num_mini_batches,
                        args.num_micro_batches,
                    )
                    self.aux_loss_stats[loss_name] = torch.zeros(stats_shape, device=device)

                self.aux_loss_stats[loss_name][
                    ppo_epoch_idx, minibatch_idx, microbatch_idx
                ] = loss_value

            # Update total auxiliary loss stats
            self.total_aux_loss_unscaled_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                total_aux_loss_unscaled
            )
            self.total_aux_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = total_aux_loss

    def _get_train_metrics(self):
        """Collect training metrics including auxiliary loss averages.

        Calls the parent method for standard PPO metrics, then appends:

        * ``"loss/aux_{name}_avg"`` – mean of each individual auxiliary loss
          across all PPO epochs, mini-batches, and micro-batches.
        * ``"loss/total_aux_loss_unscaled_avg"`` – mean of the
          coefficient-weighted sum before global scaling.
        * ``"loss/total_aux_loss_avg"`` – mean of the fully scaled total
          auxiliary loss.
        * ``"aux_loss_scale"`` – the configured global scale factor.

        Returns:
            Dict of metric name → scalar value for the completed training
            iteration, suitable for logging to W&B or TensorBoard.
        """
        metrics = super()._get_train_metrics()

        # Add auxiliary loss metrics if enabled
        if self.compute_aux_loss:
            # Add metrics for each individual auxiliary loss
            for loss_name, loss_stats in self.aux_loss_stats.items():
                metrics[f"loss/aux_{loss_name}_avg"] = (
                    self.accelerator.gather_for_metrics(loss_stats).mean().item()
                )

            # Add total auxiliary loss metrics
            metrics["loss/total_aux_loss_unscaled_avg"] = (
                self.accelerator.gather_for_metrics(self.total_aux_loss_unscaled_stats)
                .mean()
                .item()
            )
            metrics["loss/total_aux_loss_avg"] = (
                self.accelerator.gather_for_metrics(self.total_aux_loss_stats).mean().item()
            )
            metrics["aux_loss_scale"] = self.aux_loss_scale

        return metrics
