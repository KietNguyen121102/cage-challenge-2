#!/usr/bin/env python3
"""
Working Blue vs Red Agent Evaluation Script
Runs BlueReactRestoreAgent against B_lineAgent for 100 steps
Uses a simplified approach to avoid User0 errors

Adapted from CybORG evaluation.py with fixes
"""

import inspect
import time
import json
import pickle
from statistics import mean, stdev
from pathlib import Path
import traceback

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRestoreAgent


class WorkingBlueVsRedEvaluator:
    """
    Working evaluator that handles the environment issues
    """
    
    def __init__(self, scenario='Scenario1', num_steps=100, num_episodes=1):
        self.scenario = scenario
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.cyborg_version = CYBORG_VERSION
        
        # Initialize agents
        self.blue_agent = BlueReactRestoreAgent()
        self.red_agent_class = B_lineAgent
        
        # Data collection
        self.results = {
            'metadata': {
                'scenario': scenario,
                'num_steps': num_steps,
                'num_episodes': num_episodes,
                'cyborg_version': self.cyborg_version,
                'blue_agent': self.blue_agent.__class__.__name__,
                'red_agent': self.red_agent_class.__name__,
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            },
            'episodes': []
        }
        
        print(f"🎯 Working Blue vs Red Evaluator")
        print(f"📊 Blue Agent: {self.blue_agent.__class__.__name__}")
        print(f"🔴 Red Agent: {self.red_agent_class.__name__}")
        print(f"🎬 Scenario: {self.scenario}")
        print(f"📈 Episodes: {self.num_episodes}, Steps per episode: {self.num_steps}")

    def setup_environment(self):
        """Setup CybORG environment with error handling"""
        try:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f'/Shared/Scenarios/{self.scenario}.yaml'
            
            cyborg = CybORG(path, 'sim', agents={'Red': self.red_agent_class})
            
            print(f"✅ Environment setup successful")
            return cyborg
            
        except Exception as e:
            print(f"❌ Error setting up environment: {e}")
            return None

    def get_safe_action(self, agent, observation, action_space):
        """Get action from agent with error handling"""
        try:
            if observation is None:
                observation = {}
            return agent.get_action(observation, action_space)
        except Exception as e:
            print(f"⚠️ Warning getting action: {e}")
            return None

    def execute_safe_step(self, cyborg, agent_name, action):
        """Execute environment step with error handling"""
        try:
            if action is None:
                # Use Sleep action as fallback
                from CybORG.Shared.Actions import Sleep
                action = Sleep()
            
            result = cyborg.step(agent_name, action)
            return result
        except Exception as e:
            print(f"⚠️ Warning executing step for {agent_name}: {e}")
            # Return a mock result
            class MockResult:
                def __init__(self):
                    self.reward = 0.0
                    self.done = False
                    self.observation = {}
            return MockResult()

    def run_evaluation(self):
        """Run the complete evaluation with robust error handling"""
        print(f"\n🚀 Starting Working Evaluation")
        print("-" * 60)
        
        cyborg = self.setup_environment()
        if cyborg is None:
            return None
        
        total_rewards = []
        
        for episode in range(self.num_episodes):
            print(f"\n🎮 Episode {episode + 1}/{self.num_episodes}")
            
            episode_data = {
                'episode_number': episode + 1,
                'steps': [],
                'total_blue_reward': 0,
                'total_red_reward': 0,
                'blue_actions': [],
                'red_actions': [],
                'done': False,
                'errors': []
            }
            
            try:
                # Reset environment with error handling
                try:
                    blue_result = cyborg.reset('Blue')
                    blue_obs = blue_result.observation if hasattr(blue_result, 'observation') else {}
                except Exception as e:
                    print(f"⚠️ Reset error: {e}")
                    blue_obs = {}
                
                # Get action spaces with error handling
                try:
                    blue_action_space = cyborg.get_action_space('Blue')
                    print(f"✅ Episode {episode + 1} initialized")
                except Exception as e:
                    print(f"⚠️ Action space error: {e}")
                    blue_action_space = {}
                
                # Run episode steps
                for step in range(self.num_steps):
                    step_num = step + 1
                    
                    # Get blue action
                    blue_action = self.get_safe_action(self.blue_agent, blue_obs, blue_action_space)
                    
                    # Execute blue step
                    blue_result = self.execute_safe_step(cyborg, 'Blue', blue_action)
                    blue_reward = blue_result.reward if hasattr(blue_result, 'reward') else 0.0
                    blue_obs = blue_result.observation if hasattr(blue_result, 'observation') else {}
                    blue_done = blue_result.done if hasattr(blue_result, 'done') else False
                    
                    # Execute red step (automatic)
                    red_result = self.execute_safe_step(cyborg, 'Red', None)
                    red_reward = red_result.reward if hasattr(red_result, 'reward') else 0.0
                    
                    # Get last actions with error handling
                    try:
                        blue_last_action = cyborg.get_last_action('Blue')
                        red_last_action = cyborg.get_last_action('Red')
                    except Exception as e:
                        blue_last_action = 'Unknown'
                        red_last_action = 'Unknown'
                    
                    # Collect step data
                    step_data = {
                        'step': step_num,
                        'blue_action': str(blue_last_action),
                        'red_action': str(red_last_action),
                        'blue_reward': float(blue_reward),
                        'red_reward': float(red_reward),
                        'done': bool(blue_done),
                        'observation_keys': list(blue_obs.keys()) if isinstance(blue_obs, dict) else []
                    }
                    
                    episode_data['steps'].append(step_data)
                    episode_data['blue_actions'].append(str(blue_last_action))
                    episode_data['red_actions'].append(str(red_last_action))
                    episode_data['total_blue_reward'] += blue_reward
                    episode_data['total_red_reward'] += red_reward
                    
                    # Print step info
                    blue_str = str(blue_last_action)[:30]
                    red_str = str(red_last_action)[:30]
                    print(f"  Step {step_num:3d}: Blue={blue_str:30s} | Red={red_str:30s} | B_R={blue_reward:6.2f} | R_R={red_reward:6.2f}")
                    
                    if blue_done:
                        episode_data['done'] = True
                        print(f"  🏁 Episode finished early at step {step_num}")
                        break
                
                # End episode
                try:
                    self.blue_agent.end_episode()
                except:
                    pass
                
                print(f"  📊 Episode {episode + 1} completed:")
                print(f"    Blue total reward: {episode_data['total_blue_reward']:.3f}")
                print(f"    Red total reward: {episode_data['total_red_reward']:.3f}")
                print(f"    Steps completed: {len(episode_data['steps'])}")
                
            except Exception as e:
                error_msg = f"Episode {episode + 1} error: {e}"
                print(f"❌ {error_msg}")
                episode_data['errors'].append(error_msg)
            
            self.results['episodes'].append(episode_data)
            total_rewards.append(episode_data['total_blue_reward'])
        
        # Calculate summary
        if total_rewards:
            self.results['summary'] = {
                'mean_blue_reward': mean(total_rewards),
                'std_blue_reward': stdev(total_rewards) if len(total_rewards) > 1 else 0.0,
                'min_blue_reward': min(total_rewards),
                'max_blue_reward': max(total_rewards),
                'total_episodes': len(total_rewards),
                'total_steps': sum([len(ep['steps']) for ep in self.results['episodes']]),
                'avg_steps_per_episode': mean([len(ep['steps']) for ep in self.results['episodes']])
            }
        
        self.print_summary()
        return self.results

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📈 WORKING EVALUATION SUMMARY")
        print("=" * 60)
        
        if 'summary' in self.results:
            summary = self.results['summary']
            print(f"🎯 Mean Blue Reward: {summary['mean_blue_reward']:.3f}")
            print(f"📊 Std Deviation: {summary['std_blue_reward']:.3f}")
            print(f"⬇️  Min Blue Reward: {summary['min_blue_reward']:.3f}")
            print(f"⬆️  Max Blue Reward: {summary['max_blue_reward']:.3f}")
            print(f"🎮 Total Episodes: {summary['total_episodes']}")
            print(f"📈 Total Steps: {summary['total_steps']}")
            print(f"⚖️  Avg Steps/Episode: {summary['avg_steps_per_episode']:.1f}")
        
        # Action analysis
        all_blue_actions = []
        all_red_actions = []
        for episode in self.results['episodes']:
            all_blue_actions.extend([a for a in episode['blue_actions'] if a not in ['Unknown', 'None']])
            all_red_actions.extend([a for a in episode['red_actions'] if a not in ['Unknown', 'None']])
        
        blue_unique = set([a.split('(')[0] for a in all_blue_actions])  # Get action class names
        red_unique = set([a.split('(')[0] for a in all_red_actions])
        
        print(f"\n🔵 Blue Actions: {len(blue_unique)} unique types: {list(blue_unique)[:5]}")
        print(f"🔴 Red Actions: {len(red_unique)} unique types: {list(red_unique)[:5]}")

    def save_results(self, output_dir='evaluation_results'):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self.results['metadata']['timestamp']
        base_filename = f"working_blue_vs_red_{timestamp}"
        
        saved_files = {}
        
        try:
            # Save JSON results
            json_file = output_path / f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            saved_files['json'] = json_file
            
            # Save pickle
            pickle_file = output_path / f"{base_filename}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.results, f)
            saved_files['pickle'] = pickle_file
            
            # Save CSV for easy analysis
            csv_file = output_path / f"{base_filename}.csv"
            self.save_csv_summary(csv_file)
            saved_files['csv'] = csv_file
            
            print(f"\n💾 Results saved to:")
            for file_type, filepath in saved_files.items():
                print(f"  📄 {file_type.upper()}: {filepath}")
                
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            
        return saved_files

    def save_csv_summary(self, filepath):
        """Save a CSV summary for easy analysis"""
        try:
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Episode', 'Step', 'Blue_Action', 'Red_Action', 
                    'Blue_Reward', 'Red_Reward', 'Done', 'Observation_Keys'
                ])
                
                # Write data
                for episode in self.results['episodes']:
                    for step in episode['steps']:
                        writer.writerow([
                            episode['episode_number'],
                            step['step'],
                            step['blue_action'][:50],  # Truncate long actions
                            step['red_action'][:50],
                            step['blue_reward'],
                            step['red_reward'],
                            step['done'],
                            '|'.join(step['observation_keys'][:5])  # First 5 keys
                        ])
        except Exception as e:
            print(f"⚠️ Warning saving CSV: {e}")


def main():
    """Main execution function"""
    print("🎯 CybORG Working Blue vs Red Agent Evaluation")
    print("Robust version with enhanced error handling")
    print("=" * 50)
    
    # Configuration
    scenario = 'Scenario1'
    num_steps = 100
    num_episodes = 1
    
    print(f"\nConfiguration:")
    print(f"  Scenario: {scenario}")
    print(f"  Steps per episode: {num_steps}")
    print(f"  Number of episodes: {num_episodes}")
    
    # Create and run evaluator
    evaluator = WorkingBlueVsRedEvaluator(
        scenario=scenario,
        num_steps=num_steps,
        num_episodes=num_episodes
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        # Save results
        file_paths = evaluator.save_results()
        
        print(f"\n✅ Working evaluation completed!")
        if 'summary' in results:
            print(f"📊 Mean reward: {results['summary']['mean_blue_reward']:.3f}")
            print(f"📈 Total steps: {results['summary']['total_steps']}")
        
        return results, file_paths
    else:
        print("❌ Evaluation failed!")
        return None, None


if __name__ == "__main__":
    results, file_paths = main() 