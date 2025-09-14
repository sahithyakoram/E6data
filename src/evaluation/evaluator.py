from evaluation.instruction_following import score_instruction_following
from evaluation.coherence_accuracy import score_coherence_accuracy
from evaluation.hallucination_detection import score_hallucination
from evaluation.style_matching import score_style_matching
from evaluation.length_penalty import score_length_penalty
from evaluation.assumption_control import score_assumption_control



def evaluate_agent_response(agent_id: str, prompt: str, response: str, reference: str = "", mode: str = "semantic") -> dict:
    instr   = score_instruction_following(prompt, response)
    coh     = score_coherence_accuracy(response)
    halluc  = score_hallucination(response)
    style   = score_style_matching(response)
    length  = score_length_penalty(response)
    assume  = score_assumption_control(response)

    return {
        "agent_id": agent_id,
        "scores": {
            "instruction_following": instr["score"],
            "coherence_accuracy": coh["score"],
            "hallucination_detection": halluc["score"],
            "style_matching": style["score"],
            "length_penalty": length["score"],
            "assumption_control": assume["score"]
        },
        "explanations": {
            "instruction_following": instr["explanation"],
            "coherence_accuracy": coh["explanation"],
            "hallucination_detection": halluc["explanation"],
            "style_matching": style["explanation"],
            "length_penalty": length["explanation"],
            "assumption_control": assume["explanation"]
        }
    }
