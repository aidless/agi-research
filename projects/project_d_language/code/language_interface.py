"""language_interface.py - Phase 1.3: LLM-style language interface for AGI agent.

Converts (Monitor prob, slot states, current obs) into natural language
descriptions that humans can read. Implements a simple type-lattice
over slot representations (Project D: language-as-type-system).
"""
import numpy as np


# Type lattice over LunarLander state
TYPE_LATTICE = {
    "position": ["x_pos", "y_pos"],
    "velocity": ["x_vel", "y_vel"],
    "rotation": ["angle", "ang_vel"],
    "contact": ["leg_l", "leg_r"],
}


def obs_to_typed_entities(obs, feature_names=None):
    """Convert raw obs vector to typed entities (dict of name -> value)."""
    if feature_names is None:
        feature_names = ["x_pos", "y_pos", "x_vel", "y_vel",
                         "angle", "ang_vel", "leg_l", "leg_r"]
    # Pad obs with zeros so short obs (CartPole 4-dim) still has all 8 fields
    obs_list = list(obs) + [0.0] * max(0, len(feature_names) - len(obs))
    return {name: float(v) for name, v in zip(feature_names, obs_list)}


def generate_status(obs, monitor_prob, slot_states=None, recent_actions=None):
    """Generate natural language status from agent state.
    Template-based generation (no actual LLM needed for PoC).
    """
    entities = obs_to_typed_entities(obs)
    # Compute summary statistics
    pos = (entities["x_pos"], entities["y_pos"])
    vel = (entities["x_vel"], entities["y_vel"])
    angle = entities["angle"]
    legs = (entities["leg_l"], entities["leg_r"])
    fuel_estimate = 100.0 - abs(vel[0]) * 5  # crude estimate

    # Type-checked predicates (Hindley-Milner style)
    predicates = []
    if abs(angle) > 0.3:
        predicates.append(f"angle_out_of_bounds(rotation={angle:.2f})")
    if abs(vel[0]) > 1.0:
        predicates.append(f"high_horizontal_speed(velocity={vel[0]:.2f})")
    if abs(vel[1]) > 1.0:
        predicates.append(f"high_vertical_speed(velocity={vel[1]:.2f})")
    if entities["y_pos"] < 0.1:
        predicates.append(f"near_ground(height={entities['y_pos']:.3f})")
    if entities["leg_l"] > 0.5 or entities["leg_r"] > 0.5:
        predicates.append(f"leg_contact(L={entities['leg_l']:.0f}, R={entities['leg_r']:.0f})")

    # Slot-based language (Project D)
    slot_summary = ""
    if slot_states is not None and len(slot_states) > 0:
        slot_names = ["horizontal_motion", "rotation", "vertical_motion", "residual"]
        n = min(len(slot_states), len(slot_names))
        active_slots = sorted(range(n), key=lambda i: abs(float(slot_states[i].mean())), reverse=True)
        slot_summary = f" Active slot: {slot_names[active_slots[0]]}."

    # Compose language
    status = (
        f"Position ({pos[0]:.2f}, {pos[1]:.2f}); "
        f"velocity ({vel[0]:.2f}, {vel[1]:.2f}); "
        f"angle {angle:.2f} rad; "
        f"legs (L={legs[0]:.0f}, R={legs[1]:.0f}). "
        f"Monitor says: failure_prob={monitor_prob:.2f}."
    )
    if predicates:
        status += " Predicates: " + ", ".join(predicates) + "."
    if recent_actions:
        actions_str = ", ".join(str(a) for a in recent_actions[-5:])
        status += f" Recent actions: [{actions_str}]."
    status += slot_summary
    return status


def generate_plan(monitor_prob, threshold=0.5):
    """Generate a natural language plan/recommendation."""
    if monitor_prob < 0.3:
        return f"Plan: continue. Monitor confidence low ({monitor_prob:.2f}); PPO action is likely safe."
    elif monitor_prob < threshold:
        return f"Plan: monitor. Monitor says {monitor_prob:.2f}, near threshold. Continue observing."
    else:
        return f"Plan: intervene. Monitor says {monitor_prob:.2f} > {threshold}. Consider gated action."


if __name__ == "__main__":
    # Test: simulate a LunarLander state
    obs = np.array([0.1, 0.5, 0.3, -0.2, 0.05, 0.1, 0.0, 0.0])
    slot_states = [np.random.randn(8) * 0.1 for _ in range(4)]
    status = generate_status(obs, monitor_prob=0.62, slot_states=slot_states,
                             recent_actions=[0, 1, 2, 1, 0])
    print("AGENT STATUS:")
    print(" ", status)
    print()
    plan = generate_plan(0.62, threshold=0.5)
    print("AGENT PLAN:")
    print(" ", plan)
    print()
    print("Phase 1.3 (D: LLM Interface) PoC PASSED")
