#!/usr/bin/env python3
"""
Blue vs Red Agent Evaluation Script
Runs BlueReactRestoreAgent against B_lineAgent for 100 steps
Collects actions, observations, and rewards for analysis

Adapted from CybORG evaluation.py structure
"""

import subprocess
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
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent


def get_git_revision_hash():
    """Get current git commit hash"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return 'unknown'


class BlueVsRedEvaluator:
    """
    Evaluator for Blue vs Red agent scenarios
    Adapted from CybORG evaluation.py structure
    """
    
    def __init__(self, scenario='Scenario1', num_steps=100, num_episodes=1, use_wrapper=False):
        self.scenario = scenario
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.use_wrapper = use_wrapper
        self.cyborg_version = CYBORG_VERSION
        self.commit_hash = get_git_revision_hash()
        
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
                'commit_hash': self.commit_hash,
                'blue_agent': self.blue_agent.__class__.__name__,
                'red_agent': self.red_agent_class.__name__,
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'use_wrapper': use_wrapper
            },
            'episodes': []
        }
        
        print(f"🎯 Blue vs Red Evaluator Initialized")
        print(f"📊 Blue Agent: {self.blue_agent.__class__.__name__}")
        print(f"🔴 Red Agent: {self.red_agent_class.__name__}")
        print(f"🎬 Scenario: {self.scenario}")
        print(f"📈 Episodes: {self.num_episodes}, Steps per episode: {self.num_steps}")
        print(f"🔧 CybORG version: {self.cyborg_version}")

    def setup_environment(self):
        """Setup the CybORG environment"""
        try:
            # Get scenario path
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f'/Shared/Scenarios/{self.scenario}.yaml'
            
            # Initialize CybORG with red agent
            cyborg = CybORG(path, 'sim', agents={'Red': self.red_agent_class})
            
            print(f"✅ Environment setup successful")
            print(f"📁 Scenario path: {path}")
            
            return cyborg
            
        except Exception as e:
            print(f"❌ Error setting up environment: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return None

    def run_single_episode(self, episode_num, cyborg):
        """Run a single episode of the evaluation"""
        print(f"\n🎮 Episode {episode_num}/{self.num_episodes}")
        
        # Episode data collection
        episode_data = {
            'episode_number': episode_num,
            'steps': [],
            'total_blue_reward': 0,
            'total_red_reward': 0,
            'blue_actions': [],
            'red_actions': [],
            'observations': [],
            'rewards': [],
            'done': False,
            'errors': []
        }
        
        try:
            # Reset environment
            blue_result = cyborg.reset('Blue')
            red_result = cyborg.reset('Red')
            
            # Get initial action spaces
            blue_action_space = cyborg.get_action_space('Blue')
            red_action_space = cyborg.get_action_space('Red')
            
            print(f"✅ Episode {episode_num} initialized")
            print(f"   Blue action space size: {len(blue_action_space) if isinstance(blue_action_space, dict) else blue_action_space}")
            
            # Get initial observation
            current_observation = blue_result.observation if hasattr(blue_result, 'observation') else {}
            
            # Run episode steps
            for step in range(self.num_steps):
                step_data = self.run_single_step(
                    step + 1, 
                    cyborg, 
                    current_observation, 
                    blue_action_space
                )
                
                if step_data is None:
                    break
                    
                episode_data['steps'].append(step_data)
                episode_data['blue_actions'].append(step_data['blue_action'])
                episode_data['red_actions'].append(step_data['red_action'])
                episode_data['rewards'].append(step_data['blue_reward'])
                episode_data['total_blue_reward'] += step_data['blue_reward']
                episode_data['total_red_reward'] += step_data.get('red_reward', 0)
                
                # Update current observation for next step
                current_observation = step_data.get('observation', {})
                
                # Check if episode should end
                if step_data.get('done', False):
                    episode_data['done'] = True
                    print(f"  🏁 Episode {episode_num} finished early at step {step + 1}")
                    break
            
            # End episode for blue agent
            self.blue_agent.end_episode()
            
            print(f"  📊 Episode {episode_num} completed:")
            print(f"    Blue total reward: {episode_data['total_blue_reward']:.3f}")
            print(f"    Red total reward: {episode_data['total_red_reward']:.3f}")
            print(f"    Steps completed: {len(episode_data['steps'])}")
            
        except Exception as e:
            error_msg = f"Error in episode {episode_num}: {e}"
            print(f"❌ {error_msg}")
            episode_data['errors'].append(error_msg)
            
        return episode_data

    def run_single_step(self, step_num, cyborg, observation, action_space):
        """Run a single step of the simulation"""
        try:
            # Get blue action
            blue_action = self.blue_agent.get_action(observation, action_space)
            
            # Execute blue step
            blue_result = cyborg.step('Blue', blue_action)
            blue_reward = blue_result.reward if hasattr(blue_result, 'reward') else 0
            blue_done = blue_result.done if hasattr(blue_result, 'done') else False
            blue_observation = blue_result.observation if hasattr(blue_result, 'observation') else {}
            
            # Execute red step (automatic)
            red_result = cyborg.step('Red', None)
            red_reward = red_result.reward if hasattr(red_result, 'reward') else 0
            
            # Get last actions from both agents
            blue_last_action = cyborg.get_last_action('Blue')
            red_last_action = cyborg.get_last_action('Red')
            
            # Create step data
            step_data = {
                'step': step_num,
                'blue_action': str(blue_last_action) if blue_last_action else 'None',
                'red_action': str(red_last_action) if red_last_action else 'None',
                'blue_reward': float(blue_reward),
                'red_reward': float(red_reward),
                'done': bool(blue_done),
                'observation': blue_observation,
                'observation_keys': list(blue_observation.keys()) if isinstance(blue_observation, dict) else [],
                'observation_summary': self.summarize_observation(blue_observation)
            }
            
            # Print step summary
            blue_action_str = str(blue_last_action)[:40] if blue_last_action else 'None'
            red_action_str = str(red_last_action)[:40] if red_last_action else 'None'
            print(f"  Step {step_num:3d}: Blue={blue_action_str:40s} | Red={red_action_str:40s} | B_R={blue_reward:6.2f} | R_R={red_reward:6.2f}")
            
            return step_data
            
        except Exception as e:
            print(f"❌ Error in step {step_num}: {e}")
            return None

    def summarize_observation(self, observation):
        """Create a summary of the observation to avoid huge data structures"""
        if not isinstance(observation, dict):
            return str(type(observation))
            
        summary = {}
        for key, value in observation.items():
            if isinstance(value, dict):
                summary[key] = f"dict({len(value)} items)"
            elif isinstance(value, list):
                summary[key] = f"list({len(value)} items)"
            else:
                summary[key] = str(type(value))
        return summary

    def run_evaluation(self):
        """Run the complete evaluation"""
        print(f"\n🚀 Starting Blue vs Red Evaluation")
        print("-" * 60)
        
        # Setup environment
        cyborg = self.setup_environment()
        if cyborg is None:
            print("❌ Failed to setup environment")
            return None
        
        total_rewards = []
        
        # Run episodes
        for episode in range(self.num_episodes):
            episode_data = self.run_single_episode(episode + 1, cyborg)
            self.results['episodes'].append(episode_data)
            total_rewards.append(episode_data['total_blue_reward'])
        
        # Calculate summary statistics
        if total_rewards:
            self.results['summary'] = {
                'mean_blue_reward': mean(total_rewards),
                'std_blue_reward': stdev(total_rewards) if len(total_rewards) > 1 else 0.0,
                'min_blue_reward': min(total_rewards),
                'max_blue_reward': max(total_rewards),
                'total_episodes': len(total_rewards),
                'avg_steps_per_episode': mean([len(ep['steps']) for ep in self.results['episodes']]),
                'total_steps': sum([len(ep['steps']) for ep in self.results['episodes']])
            }
        
        self.print_summary()
        return self.results

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📈 EVALUATION SUMMARY")
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
            all_blue_actions.extend([a for a in episode['blue_actions'] if a != 'None'])
            all_red_actions.extend([a for a in episode['red_actions'] if a != 'None'])
        
        print(f"\n🔵 Blue Actions: {len(set(all_blue_actions))} unique types")
        print(f"🔴 Red Actions: {len(set(all_red_actions))} unique types")

    def save_results(self, output_dir='evaluation_results'):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self.results['metadata']['timestamp']
        base_filename = f"blue_vs_red_{timestamp}"
        
        saved_files = {}
        
        try:
            # Save JSON results (without large observations)
            json_data = self.create_json_safe_results()
            json_file = output_path / f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            saved_files['json'] = json_file
            
            # Save pickle for complete Python objects
            pickle_file = output_path / f"{base_filename}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.results, f)
            saved_files['pickle'] = pickle_file
            
            # Save summary text
            txt_file = output_path / f"{base_filename}_summary.txt"
            self.save_text_summary(txt_file)
            saved_files['summary'] = txt_file
            
            print(f"\n💾 Results saved to:")
            for file_type, filepath in saved_files.items():
                print(f"  📄 {file_type.upper()}: {filepath}")
                
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            
        return saved_files

    def create_json_safe_results(self):
        """Create a JSON-safe version of results (without large observation objects)"""
        json_safe = {
            'metadata': self.results['metadata'],
            'summary': self.results.get('summary', {}),
            'episodes': []
        }
        
        for episode in self.results['episodes']:
            safe_episode = {
                'episode_number': episode['episode_number'],
                'total_blue_reward': episode['total_blue_reward'],
                'total_red_reward': episode['total_red_reward'],
                'done': episode['done'],
                'errors': episode['errors'],
                'steps': []
            }
            
            for step in episode['steps']:
                safe_step = {
                    'step': step['step'],
                    'blue_action': step['blue_action'],
                    'red_action': step['red_action'],
                    'blue_reward': step['blue_reward'],
                    'red_reward': step['red_reward'],
                    'done': step['done'],
                    'observation_keys': step['observation_keys'],
                    'observation_summary': step['observation_summary']
                }
                safe_episode['steps'].append(safe_step)
            
            json_safe['episodes'].append(safe_episode)
            
        return json_safe

    def save_text_summary(self, filepath):
        """Save a human-readable text summary"""
        with open(filepath, 'w') as f:
            f.write("Blue vs Red Agent Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            f.write("Metadata:\n")
            for key, value in self.results['metadata'].items():
                f.write(f"  {key}: {value}\n")
            
            # Summary statistics
            if 'summary' in self.results:
                f.write("\nSummary Statistics:\n")
                for key, value in self.results['summary'].items():
                    f.write(f"  {key}: {value}\n")
            
            # Episode details
            f.write("\nEpisode Details:\n")
            for episode in self.results['episodes']:
                f.write(f"\nEpisode {episode['episode_number']}:\n")
                f.write(f"  Blue Total Reward: {episode['total_blue_reward']:.3f}\n")
                f.write(f"  Red Total Reward: {episode['total_red_reward']:.3f}\n")
                f.write(f"  Steps Completed: {len(episode['steps'])}\n")
                f.write(f"  Episode Done: {episode['done']}\n")
                if episode['errors']:
                    f.write(f"  Errors: {len(episode['errors'])}\n")


def main():
    """Main execution function - adapted from evaluation.py structure"""
    print("🎯 CybORG Blue vs Red Agent Evaluation")
    print("Adapted from evaluation.py structure")
    print("=" * 50)
    
    # Configuration
    scenario = 'Scenario1'  # Can be changed to Scenario1b or Scenario2
    num_steps = 100
    num_episodes = 1
    
    # Ask for evaluation details (similar to evaluation.py)
    print(f"\nEvaluation Configuration:")
    print(f"  Scenario: {scenario}")
    print(f"  Steps per episode: {num_steps}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Blue Agent: BlueReactRestoreAgent")
    print(f"  Red Agent: B_lineAgent")
    
    # Create and run evaluator
    evaluator = BlueVsRedEvaluator(
        scenario=scenario,
        num_steps=num_steps,
        num_episodes=num_episodes
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        # Save results
        file_paths = evaluator.save_results()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Total reward: {results.get('summary', {}).get('mean_blue_reward', 'N/A')}")
        
        return results, file_paths
    else:
        print("❌ Evaluation failed!")
        return None, None


if __name__ == "__main__":
    results, file_paths = main() 