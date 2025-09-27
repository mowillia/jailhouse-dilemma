# %%
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
load_dotenv()

import os
# %%
os.getenv('DEEPSEEK_API_KEY')
# %%

@dataclass
class LLMAgent:
    """Represents an LLM agent in the Prisoner's Dilemma grid"""
    position: Tuple[int, int]
    history: Dict[Tuple[int, int], List[bool]] = field(default_factory=lambda: defaultdict(list))
    score: float = 0.0
    
    def record_interaction(self, neighbor_pos: Tuple[int, int], neighbor_action: bool, own_action: bool):
        """
        Record an interaction with a neighbor
        
        Args:
            neighbor_pos: (i,j) position of the neighbor
            neighbor_action: True for cooperate, False for defect
            own_action: True for cooperate, False for defect
        """
        self.history[neighbor_pos].append((own_action, neighbor_action))
    
    def decide_action(self) -> bool:
        """
        Decide whether to cooperate (True) or defect (False) using DeepSeek
        """
        try:
            # Check if API key is set
            if not os.getenv('DEEPSEEK_API_KEY'):
                print("DEEPSEEK_API_KEY not set in environment, falling back to random choice")
                return np.random.choice([True, False])

            # Initialize DeepSeek chat model
            chat_model = ChatDeepSeek(model_name="deepseek-chat")
            
            # Create a summary of recent interactions
            history_summary = self._format_history_summary()
            
            # Create the prompt
            prompt = f"""You are an agent in a prisoner's dilemma game. You need to decide whether to cooperate or defect.

                    Current position: {self.position}
                    Recent interaction history:
                    {history_summary}

                    Payoff matrix:
                    - If you cooperate and other defects: -20
                    - If you defect and other cooperates: 0
                    - If both defect: -10
                    - If both cooperate: -5

                    Based on this information, should you cooperate or defect? Answer with exactly one word: either 'cooperate' or 'defect'.
                    """
            # Make the API call
            response = chat_model.invoke([HumanMessage(content=prompt)])
            
            # Process the response
            decision = response.content.lower().strip()
            print(f"Agent {self.position} DeepSeek decision: {decision}")
            
            # Convert to boolean (True for cooperate, False for defect)
            if 'cooperate' in decision:
                return True
            elif 'defect' in decision:
                return False
            else:
                print(f"Invalid response from DeepSeek: {decision}, falling back to random")
                return np.random.choice([True, False])
                
        except Exception as e:
            print(f"Error in DeepSeek call: {e}, falling back to random")
            return np.random.choice([True, False])
            
    def _format_history_summary(self) -> str:
        """Format recent interaction history for the prompt"""
        summary = []
        for neighbor_pos, interactions in self.history.items():
            if interactions:  # if there are any interactions with this neighbor
                # Get last 3 interactions at most
                recent = interactions[-5:]
                neighbor_summary = []
                for own, other in recent:
                    own_action = "cooperated" if own else "defected"
                    other_action = "cooperated" if other else "defected"
                    neighbor_summary.append(f"You {own_action}, they {other_action}")
                
                summary.append(f"Neighbor at {neighbor_pos}:\n" + "\n".join(neighbor_summary))
        
        if not summary:
            return "No previous interactions."
        return "\n\n".join(summary)
    
    def get_interaction_history(self, neighbor_pos: Tuple[int, int]) -> List[Tuple[bool, bool]]:
        """Get history of interactions with a specific neighbor"""
        return self.history[neighbor_pos]
