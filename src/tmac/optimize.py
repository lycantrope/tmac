import jax
import optax

_OPT_LBFGS = optax.lbfgs(
    linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=30)
)


def _lbfgs_minimize(loss_fn, x0, n_steps):
    """Minimize loss_fn from x0 with a fixed number of L-BFGS steps."""
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def step(carry, _):
        x, state = carry
        value, grad = value_and_grad(x, state=state)
        updates, state = _OPT_LBFGS.update(
            grad, state, x, value=value, grad=grad, value_fn=loss_fn
        )
        return (optax.apply_updates(x, updates), state), None

    (x, _), _ = jax.lax.scan(step, (x0, _OPT_LBFGS.init(x0)), None, length=n_steps)
    return x
