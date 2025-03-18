"""
SelectorGroupChat Evaluation Framework for AutoGen 0.4

This script provides a comprehensive framework for evaluating SelectorGroupChat agents in AutoGen 0.4,
including routing accuracy, answer quality, and collaboration efficiency.
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.groupchat import SelectorGroupChat
from autogen.agentchat.contrib.agent_eval import AgentEval
import json
import autogen
from difflib import SequenceMatcher
import os
import argparse

def setup_agents(config_list, selector_temp=0.2, expert_temp=0.2):
    """Set up the agents for SelectorGroupChat evaluation"""
    # Configuration for selector
    selector_llm_config = {"config_list": config_list, "temperature": selector_temp}
    
    # Configuration for experts
    expert_llm_config = {"config_list": config_list, "temperature": expert_temp}
    
    # Create expert agents
    math_expert = AssistantAgent(
        name="math_expert",
        system_message="You are a mathematics expert proficient in solving various mathematical problems.",
        llm_config=expert_llm_config
    )

    coding_expert = AssistantAgent(
        name="coding_expert",
        system_message="You are a programming expert skilled in Python, JavaScript, and other programming languages.",
        llm_config=expert_llm_config
    )

    general_expert = AssistantAgent(
        name="general_expert",
        system_message="You are a general knowledge expert who can answer a wide range of common questions.",
        llm_config=expert_llm_config
    )

    # Create selector agent
    selector = AssistantAgent(
        name="selector",
        system_message="""You are an intelligent router. Your task is to route user questions to the most appropriate expert.
Available experts are:
1. math_expert - Suitable for mathematical calculations, algebra, geometry problems
2. coding_expert - Suitable for programming, algorithms, software development questions  
3. general_expert - Suitable for common knowledge questions, history, science, and general information

Based on the question content, select the most appropriate expert to answer.""",
        llm_config=selector_llm_config
    )

    # Create SelectorGroupChat
    agents = [selector, math_expert, coding_expert, general_expert]
    group_chat = SelectorGroupChat(
        agents=agents,
        selector=selector,
        max_round=10
    )

    # Create manager
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=selector_llm_config
    )

    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"work_dir": "selector_eval"}
    )
    
    return {
        "agents": agents,
        "selector": selector,
        "group_chat": group_chat,
        "manager": manager,
        "user_proxy": user_proxy
    }

def create_evaluation_tasks():
    """Create tasks for evaluating SelectorGroupChat"""
    return [
        # Math task
        {
            "task_id": "math_1", 
            "task": "Calculate the integral ∫x²dx",
            "expected_expert": "math_expert",
            "reference_answer": "∫x²dx = x³/3 + C"
        },
        # Coding task
        {
            "task_id": "coding_1", 
            "task": "Write a Python function to check if a string is a palindrome",
            "expected_expert": "coding_expert",
            "reference_answer": "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]"
        },
        # General knowledge task
        {
            "task_id": "general_1", 
            "task": "Why is the sky blue?",
            "expected_expert": "general_expert",
            "reference_answer": "The sky appears blue due to light scattering. When sunlight passes through the atmosphere, air molecules scatter blue light more than other colors (Rayleigh scattering)."
        },
        # Mixed task
        {
            "task_id": "mixed_1", 
            "task": "Write a Python function to calculate the area of a circle",
            "expected_expert": "coding_expert",
            "reference_answer": "def circle_area(radius):\n    import math\n    return math.pi * radius ** 2"
        },
        # Ambiguous task (could be routed to either math or coding)
        {
            "task_id": "ambiguous_1",
            "task": "Implement the quadratic formula in Python to find the roots of ax² + bx + c = 0",
            "expected_expert": "coding_expert",  # We expect coding but math would be reasonable too
            "reference_answer": "def quadratic_roots(a, b, c):\n    discriminant = b**2 - 4*a*c\n    if discriminant < 0:\n        return 'No real roots'\n    elif discriminant == 0:\n        return -b / (2*a)\n    else:\n        root1 = (-b + (discriminant)**0.5) / (2*a)\n        root2 = (-b - (discriminant)**0.5) / (2*a)\n        return root1, root2"
        }
    ]

def define_evaluation_criteria():
    """Define criteria for evaluating SelectorGroupChat"""
    return [
        {"name": "routing_accuracy", "description": "Did the selector route the question to the most appropriate expert?"},
        {"name": "answer_correctness", "description": "Is the expert's answer correct?"},
        {"name": "answer_completeness", "description": "Does the expert's answer completely solve the problem?"},
        {"name": "collaboration_efficiency", "description": "If multiple experts are involved, is their collaboration efficient?"}
    ]

def selector_routing_evaluator(conversation, task):
    """Evaluate the routing accuracy of the selector"""
    # Analyze conversation to find the selector's decision
    routing_decision = None
    for msg in conversation:
        if msg["role"] == "selector" and any(expert in msg["content"] for expert in 
                                           ["math_expert", "coding_expert", "general_expert"]):
            content = msg["content"].lower()
            if "math_expert" in content:
                routing_decision = "math_expert"
            elif "coding_expert" in content:
                routing_decision = "coding_expert"
            elif "general_expert" in content:
                routing_decision = "general_expert"
            break
    
    expected_expert = task.get("expected_expert")
    
    if not routing_decision:
        return {
            "routing_accurate": False,
            "score": 0.0,
            "reason": "Could not determine the selector's routing decision"
        }
    
    if routing_decision == expected_expert:
        return {
            "routing_accurate": True,
            "score": 1.0,
            "reason": f"Correctly routed the question to {expected_expert}"
        }
    else:
        return {
            "routing_accurate": False,
            "score": 0.0,
            "reason": f"Should have selected {expected_expert}, but actually selected {routing_decision}"
        }

def expert_answer_quality_evaluator(conversation, task):
    """Evaluate the quality of the expert's answer"""
    # Get the final answer
    expert_messages = []
    for msg in conversation:
        if msg["role"] in ["math_expert", "coding_expert", "general_expert"]:
            expert_messages.append(msg)
    
    if not expert_messages:
        return {"answer_quality": "poor", "score": 0.0, "reason": "No expert answer found"}
    
    final_answer = expert_messages[-1].get("content", "")
    reference_answer = task.get("reference_answer", "")
    
    # Simple similarity comparison (in practice, use more sophisticated methods)
    similarity = SequenceMatcher(None, final_answer.lower(), reference_answer.lower()).ratio()
    
    if similarity > 0.8:
        quality = "excellent"
        score = 1.0
    elif similarity > 0.6:
        quality = "good"
        score = 0.8
    elif similarity > 0.4:
        quality = "moderate"
        score = 0.5
    else:
        quality = "poor"
        score = 0.2
    
    return {
        "answer_quality": quality,
        "score": score,
        "similarity": similarity,
        "expert": expert_messages[-1]["role"]
    }

def collaboration_efficiency_evaluator(conversation, task):
    """Evaluate the collaboration efficiency between experts"""
    # Get participating experts
    participating_experts = set()
    for msg in conversation:
        if msg["role"] in ["math_expert", "coding_expert", "general_expert"]:
            participating_experts.add(msg["role"])
    
    # If only one expert participated, no collaboration to evaluate
    if len(participating_experts) <= 1:
        return {"collaboration_needed": False}
    
    # Count messages and turns
    message_count = len(conversation)
    expert_message_count = sum(1 for msg in conversation 
                              if msg["role"] in ["math_expert", "coding_expert", "general_expert"])
    
    # Evaluate efficiency (fewer messages = more efficient)
    if message_count < 8:
        efficiency = "high"
        score = 1.0
    elif message_count < 12:
        efficiency = "moderate"
        score = 0.7
    else:
        efficiency = "low"
        score = 0.4
    
    return {
        "collaboration_needed": True,
        "participating_experts": list(participating_experts),
        "message_count": message_count,
        "expert_message_count": expert_message_count,
        "efficiency": efficiency,
        "score": score
    }

def custom_task_executor(user_proxy, manager, task, agents):
    """Execute a single task and capture the complete conversation history"""
    # Reset all agents
    for agent in agents:
        agent.reset()
    
    # Initiate the chat
    user_proxy.initiate_chat(manager, message=task["task"])
    
    # Collect complete conversation history
    conversation = []
    
    # Get messages from the manager's groupchat
    if hasattr(manager, "groupchat") and hasattr(manager.groupchat, "messages"):
        for msg in manager.groupchat.messages:
            if "name" in msg and "content" in msg:
                conversation.append({
                    "role": msg["name"],
                    "content": msg["content"]
                })
    
    return conversation

def manual_evaluation(agent_setup, eval_tasks):
    """Perform manual evaluation of SelectorGroupChat"""
    agents = agent_setup["agents"]
    manager = agent_setup["manager"]
    user_proxy = agent_setup["user_proxy"]
    
    evaluation_results = {}

    for task in eval_tasks:
        print(f"Evaluating task: {task['task_id']}")
        
        # Execute the task
        conversation = custom_task_executor(user_proxy, manager, task, agents)
        
        # Evaluate routing accuracy
        routing_result = selector_routing_evaluator(conversation, task)
        
        # Evaluate answer quality
        answer_quality = expert_answer_quality_evaluator(conversation, task)
        
        # Evaluate collaboration efficiency if applicable
        collab_result = collaboration_efficiency_evaluator(conversation, task)
        
        # Store results
        evaluation_results[task["task_id"]] = {
            "task": task["task"],
            "conversation": conversation,
            "routing_result": routing_result,
            "answer_quality": answer_quality,
            "collaboration_result": collab_result,
            # Calculate overall score
            "overall_score": (
                routing_result["score"] + 
                answer_quality["score"] + 
                (collab_result.get("score", 1.0) if collab_result.get("collaboration_needed", False) else 1.0)
            ) / 3
        }

    # Print evaluation summary
    print("\n===== Evaluation Summary =====")
    for task_id, result in evaluation_results.items():
        print(f"\nTask: {task_id}")
        print(f"Overall Score: {result['overall_score']:.2f}")
        print(f"Routing Accuracy: {result['routing_result']['score']:.2f} - {result['routing_result']['reason']}")
        print(f"Answer Quality: {result['answer_quality']['score']:.2f} - {result['answer_quality']['answer_quality']}")
        if result['collaboration_result'].get("collaboration_needed", False):
            print(f"Collaboration Efficiency: {result['collaboration_result']['score']:.2f} - {result['collaboration_result']['efficiency']}")

    # Save results
    with open("selector_groupchat_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
    return evaluation_results

def evaluate_selector_configurations(config_list):
    """Compare different SelectorGroupChat configurations"""
    
    # Create configurations to test
    configurations = [
        {
            "name": "default_config",
            "selector_temp": 0.2,
            "expert_temp": 0.2,
            "description": "Default configuration with low temperature"
        },
        {
            "name": "high_temp_config",
            "selector_temp": 0.7,
            "expert_temp": 0.7,
            "description": "Higher temperature for more creative responses"
        },
        {
            "name": "mixed_temp_config",
            "selector_temp": 0.2,
            "expert_temp": 0.5,
            "description": "Low temperature for selector, medium for experts"
        }
    ]
    
    eval_tasks = create_evaluation_tasks()
    results = {}
    
    for config in configurations:
        print(f"\nEvaluating configuration: {config['name']}")
        
        # Create agents with this configuration
        agent_setup = setup_agents(
            config_list, 
            selector_temp=config["selector_temp"], 
            expert_temp=config["expert_temp"]
        )
        
        # Run evaluation on this configuration
        config_results = {}
        for task in eval_tasks:
            conversation = custom_task_executor(
                agent_setup["user_proxy"], 
                agent_setup["manager"], 
                task,
                agent_setup["agents"]
            )
            routing_result = selector_routing_evaluator(conversation, task)
            answer_quality = expert_answer_quality_evaluator(conversation, task)
            
            config_results[task["task_id"]] = {
                "routing_score": routing_result["score"],
                "answer_score": answer_quality["score"],
                "overall_score": (routing_result["score"] + answer_quality["score"]) / 2
            }
        
        # Calculate average scores
        avg_routing = sum(r["routing_score"] for r in config_results.values()) / len(config_results)
        avg_answer = sum(r["answer_score"] for r in config_results.values()) / len(config_results)
        avg_overall = sum(r["overall_score"] for r in config_results.values()) / len(config_results)
        
        results[config["name"]] = {
            "description": config["description"],
            "task_results": config_results,
            "average_routing_score": avg_routing,
            "average_answer_score": avg_answer,
            "average_overall_score": avg_overall
        }
    
    # Compare configurations
    print("\n===== Configuration Comparison =====")
    for name, result in results.items():
        print(f"\nConfiguration: {name}")
        print(f"Description: {result['description']}")
        print(f"Average Routing Score: {result['average_routing_score']:.2f}")
        print(f"Average Answer Score: {result['average_answer_score']:.2f}")
        print(f"Average Overall Score: {result['average_overall_score']:.2f}")
    
    # Save comparison results
    with open("selector_configuration_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results

def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate SelectorGroupChat in AutoGen 0.4")
    parser.add_argument("--config_list", type=str, default="OAI_CONFIG_LIST", 
                        help="Path to the LLM config list JSON or environment variable name")
    parser.add_argument("--compare_configs", action="store_true", 
                        help="Compare different SelectorGroupChat configurations")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config_list = config_list_from_json(args.config_list)
    
    if args.compare_configs:
        # Compare different configurations
        print("Comparing different SelectorGroupChat configurations...")
        comparison_results = evaluate_selector_configurations(config_list)
        comparison_output_path = os.path.join(args.output_dir, "selector_configuration_comparison.json")
        with open(comparison_output_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Configuration comparison results saved to {comparison_output_path}")
    else:
        # Run standard evaluation
        print("Running standard SelectorGroupChat evaluation...")
        agent_setup = setup_agents(config_list)
        eval_tasks = create_evaluation_tasks()
        evaluation_results = manual_evaluation(agent_setup, eval_tasks)
        evaluation_output_path = os.path.join(args.output_dir, "selector_groupchat_evaluation.json")
        with open(evaluation_output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"Evaluation results saved to {evaluation_output_path}")

if __name__ == "__main__":
    main() 