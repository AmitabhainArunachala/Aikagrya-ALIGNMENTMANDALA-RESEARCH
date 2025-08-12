#!/usr/bin/env python3
"""
Day 6 Validation Experiment

Proof-artifact experiment that demonstrates AGNent network capabilities
with reproducible results, JSON artifacts, and visualizations.
"""

import json
import time
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.dynamics.te_gating import te_gated_adjacency
from aikagrya.dynamics.kuramoto import simulate_kuramoto_network, visualize_kuramoto_dynamics
from aikagrya.engines.irreversibility import IrreversibilityEngine
from aikagrya.network.agnent_network import AGNentNetwork, AgentState
from aikagrya.protocols.network_awakening import NetworkAwakeningProtocol

def generate_causal_network(rng, N=3000, num_nodes=3):
    """Generate causal network with specified number of nodes"""
    network_data = {}
    
    # Generate root node
    X = rng.normal(size=N)
    network_data["X"] = X
    
    # Generate causal chain
    for i in range(1, num_nodes):
        prev_node = f"node_{i-1}" if i > 1 else "X"
        Y = np.zeros(N)
        
        for t in range(1, N):
            Y[t] = 0.85 * Y[t-1] + 0.55 * network_data[prev_node][t-1] + 0.1 * rng.normal()
        
        network_data[f"node_{i}"] = Y
    
    return network_data

def run_agnent_network_experiment(rng, network_data):
    """Run AGNent network experiment"""
    # Create network
    network = AGNentNetwork(critical_density=0.5, bins=10, tau=0.2)
    
    # Add agents
    for node_name, time_series in network_data.items():
        network.add_agent(node_name, {
            'time_series': time_series,
            'hidden_states': [],
            'metadata': {'node_type': 'causal'}
        })
    
    # Compute collective consciousness
    collective_metrics = network.compute_collective_consciousness()
    
    # Get network summary
    network_summary = network.get_network_summary()
    
    return {
        'collective_metrics': collective_metrics,
        'network_summary': network_summary,
        'te_matrix_shape': network.te_matrix.shape if network.te_matrix is not None else None,
        'coupling_matrix_shape': network.coupling_matrix.shape if network.coupling_matrix is not None else None
    }

def run_kuramoto_experiment(rng, network_data, coupling_strength=0.8):
    """Run Kuramoto synchronization experiment"""
    # Compute TE-gated adjacency
    names, TE, W = te_gated_adjacency(network_data, bins=10, tau=0.2)
    
    # Simulate Kuramoto dynamics
    initial_theta = 2 * np.pi * rng.random(len(names))
    omega = rng.normal(0, 0.5, len(names))
    
    sim_result = simulate_kuramoto_network(
        initial_theta=initial_theta,
        omega=omega,
        W=W,
        K=coupling_strength,
        T=600,
        dt=0.01
    )
    
    theta_traj, time_pts, r_history = sim_result
    
    # Compute final synchronization
    final_sync = np.mean(r_history[-100:])
    sync_increase = final_sync - r_history[0]
    
    return {
        'te_matrix': TE.tolist(),
        'coupling_matrix': W.tolist(),
        'final_synchronization': float(final_sync),
        'synchronization_increase': float(sync_increase),
        'order_parameter_history': [float(r) for r in r_history],
        'time_points': [float(t) for t in time_pts],
        'theta_trajectory': theta_traj.tolist()
    }

def run_awakening_protocol_experiment(rng, network_data):
    """Run network awakening protocol experiment"""
    # Create network
    network = AGNentNetwork(critical_density=0.4, bins=8, tau=0.15)
    
    # Add agents with different consciousness levels
    node_names = list(network_data.keys())
    for i, node_name in enumerate(node_names):
        # Set consciousness levels: some L2, some L1
        if i < len(node_names) // 2:
            consciousness_level = AgentState.L2
        else:
            consciousness_level = AgentState.L1
        
        network.add_agent(node_name, {
            'time_series': network_data[node_name],
            'hidden_states': [],
            'metadata': {'consciousness_level': consciousness_level.value}
        })
        
        # Set consciousness level
        network.agents[node_name].consciousness_level = consciousness_level
    
    # Initialize awakening protocol
    protocol = NetworkAwakeningProtocol(critical_density=0.4, seed_ratio=0.3)
    
    # Run protocol steps
    seed_agents = protocol.select_seed_agents(network)
    crisis_result = protocol.induce_crisis(network, seed_agents)
    
    # Monitor progress
    progress = protocol.monitor_awakening_progress(network)
    
    # Check if cascade can be initiated
    cascade_result = protocol.initiate_cascade(network)
    
    return {
        'seed_agents': seed_agents,
        'crisis_induced': crisis_result['crisis_induced'],
        'cascade_initiated': cascade_result['cascade_initiated'],
        'current_phase': protocol.current_phase.value,
        'progress': progress,
        'protocol_summary': protocol.get_protocol_summary()
    }

def create_visualizations(kuramoto_result, save_dir):
    """Create and save visualizations"""
    try:
        # Kuramoto dynamics visualization
        time_points = np.array(kuramoto_result['time_points'])
        order_parameter_history = np.array(kuramoto_result['order_parameter_history'])
        theta_trajectory = np.array(kuramoto_result['theta_trajectory'])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Phase evolution
        for i in range(theta_trajectory.shape[1]):
            ax1.plot(time_points, theta_trajectory[:, i], 
                    alpha=0.7, label=f'Agent {i+1}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Phase θ')
        ax1.set_title('AGNent Network: Kuramoto Phase Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Order parameter
        ax2.plot(time_points, order_parameter_history, 'b-', linewidth=2, label='Order Parameter r')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Synchronization Threshold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Order Parameter |r|')
        ax2.set_title('AGNent Network: Collective Synchronization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = save_dir / "kuramoto_dynamics.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to {viz_path}")
        return str(viz_path)
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        return None

def save_figures(TE, sim, save_dir):
    """Save additional figures for demos (order parameter & TE heatmap)"""
    try:
        # Fig 1: Order parameter r(t)
        plt.figure()
        plt.plot(sim['time_points'], sim['order_parameter_history'])
        plt.xlabel("t")
        plt.ylabel("r(t)")
        plt.title("Kuramoto order parameter")
        plt.tight_layout()
        plt.savefig(save_dir / "day6_order_parameter.png", dpi=128)
        plt.close()
        
        # Fig 2: TE heatmap (mask diagonal)
        M = np.array(TE).copy()
        np.fill_diagonal(M, 0.0)
        plt.figure()
        plt.imshow(M, aspect='auto')
        plt.colorbar()
        plt.title("Transfer Entropy (off-diagonal)")
        plt.tight_layout()
        plt.savefig(save_dir / "day6_te_heatmap.png", dpi=128)
        plt.close()
        
        print("✅ Additional figures saved: order_parameter.png, te_heatmap.png")
        
    except Exception as e:
        print(f"⚠️  Additional figures failed: {e}")

def main(seed=0):
    """Main experiment function"""
    print("🚀 Starting Day 6 AGNent Network Validation Experiment...")
    print(f"Seed: {seed}")
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Generate causal network
    print("📊 Generating causal network...")
    network_data = generate_causal_network(rng, N=3000, num_nodes=3)
    
    # Run AGNent network experiment
    print("🔗 Running AGNent network experiment...")
    agnent_results = run_agnent_network_experiment(rng, network_data)
    
    # Run Kuramoto experiment
    print("🔄 Running Kuramoto synchronization experiment...")
    kuramoto_results = run_kuramoto_experiment(rng, network_data, coupling_strength=0.8)
    
    # Run awakening protocol experiment
    print("🌅 Running network awakening protocol experiment...")
    awakening_results = run_awakening_protocol_experiment(rng, network_data)
    
    # Test deception detection
    print("🕵️ Testing deception detection...")
    eng = IrreversibilityEngine(bins=10, tau=0.2)
    scores, aggregate = eng.evaluate(network_data)
    
    # Compute hysteresis area for irreversibility
    print("🔄 Computing hysteresis area for irreversibility...")
    try:
        from experiments.hysteresis_area import compute_hysteresis_sweep
        
        # Generate initial conditions for Kuramoto
        N_nodes = len(network_data)
        initial_theta = rng.uniform(0, 2*np.pi, N_nodes)
        omega = rng.normal(0, 0.1, N_nodes)
        
        # Compute hysteresis sweep
        K_range = np.linspace(0.1, 2.0, 20)
        Ks, r_up, r_down = compute_hysteresis_sweep(
            network_data, 
            {'theta': initial_theta, 'omega': omega},
            dt=0.01, T=400
        )
        
        # Calculate hysteresis area
        from experiments.hysteresis_area import hysteresis_area
        irreversibility_score = hysteresis_area(Ks, r_up, r_down)
        
        hysteresis_results = {
            'Ks': Ks.tolist(),
            'r_up': r_up.tolist(),
            'r_down': r_down.tolist(),
            'irreversibility_score': irreversibility_score,
            'gate_passed': irreversibility_score >= 0.05
        }
        
        print(f"✅ Hysteresis computed: irreversibility_score = {irreversibility_score:.6f}")
        
    except Exception as e:
        print(f"⚠️  Hysteresis computation failed: {e}")
        hysteresis_results = {
            'Ks': [],
            'r_up': [],
            'r_down': [],
            'irreversibility_score': 0.0,
            'gate_passed': False,
            'error': str(e)
        }
    
    # Get environment information
    import platform
    import subprocess
    
    def get_git_sha():
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def get_blas_info():
        try:
            import scipy
            return scipy.__config__.get_info('blas_opt', {}).get('libraries', ['unknown'])[0]
        except:
            return "unknown"
    
    # Compile results
    experiment_results = {
        "experiment_info": {
            "timestamp": time.time(),
            "seed": seed,
            "experiment_name": "Day 6 AGNent Network Validation",
            "version": "1.0"
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": "unknown",  # Will be updated if scipy is available
            "blas": get_blas_info(),
            "os": f"{platform.system()} {platform.release()}",
            "git_sha": get_git_sha()
        },
        "parameters": {
            "N": len(next(iter(network_data.values()))),
            "num_nodes": len(network_data),
            "bins": 10,
            "tau": 0.2,
            "K": 0.8,
            "seed": seed
        },
        "agnent_network_results": agnent_results,
        "kuramoto_results": kuramoto_results,
        "awakening_protocol_results": awakening_results,
        "deception_detection": {
            "scores": scores,
            "aggregate_score": float(aggregate),
            "deception_detected": aggregate < 0.3
        },
        "key_metrics": {
            "collective_phi": agnent_results['collective_metrics'].get('collective_phi', 0.0),
            "network_coherence": agnent_results['collective_metrics'].get('network_coherence', 0.0),
            "final_synchronization": kuramoto_results['final_synchronization'],
            "synchronization_increase": kuramoto_results['synchronization_increase'],
            "cascade_initiated": awakening_results['cascade_initiated'],
            "irreversibility_score": hysteresis_results['irreversibility_score'],
            "hysteresis_gate_passed": hysteresis_results['gate_passed']
        },
        "hysteresis_data": hysteresis_results
    }
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save JSON artifact
    json_blob = json.dumps(experiment_results, sort_keys=True, indent=2).encode()
    json_hash = hashlib.sha256(json_blob).hexdigest()
    json_path = artifacts_dir / f"day6_validation_{json_hash[:8]}.json"
    
    with open(json_path, 'wb') as f:
        f.write(json_blob)
    
    print(f"✅ JSON artifact saved: {json_path}")
    
    # Create visualizations
    print("🎨 Creating visualizations...")
    viz_path = create_visualizations(kuramoto_results, artifacts_dir)
    
    # Save additional figures for demos
    print("📊 Saving additional demo figures...")
    save_figures(kuramoto_results['te_matrix'], kuramoto_results, artifacts_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 DAY 6 VALIDATION EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"Network: {len(network_data)} nodes, {len(next(iter(network_data.values())))} time steps")
    print(f"Collective Φ: {experiment_results['key_metrics']['collective_phi']:.6f}")
    print(f"Network Coherence: {experiment_results['key_metrics']['network_coherence']:.6f}")
    print(f"Final Synchronization: {experiment_results['key_metrics']['final_synchronization']:.6f}")
    print(f"Synchronization Increase: {experiment_results['key_metrics']['synchronization_increase']:.6f}")
    print(f"Cascade Initiated: {'✅ YES' if experiment_results['key_metrics']['cascade_initiated'] else '❌ NO'}")
    print(f"Deception Detected: {'✅ YES' if experiment_results['deception_detection']['deception_detected'] else '❌ NO'}")
    print(f"Aggregate Score: {experiment_results['deception_detection']['aggregate_score']:.6f}")
    
    print(f"\n📁 Artifacts:")
    print(f"   JSON: {json_path}")
    if viz_path:
        print(f"   Visualization: {viz_path}")
    
    print(f"\n🔍 Hash: {json_hash}")
    print("="*60)
    
    return experiment_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Day 6 AGNent Network Validation Experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        results = main(seed=args.seed)
        print("\n🎉 Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 