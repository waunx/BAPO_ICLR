import matplotlib.pyplot as plt
import numpy as np

def create_bapo_plot():
    """Create BAPO test accuracy plot with same style"""
    
    # Your BAPO data
    steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    
    
    test_accuracy_grpo = [0.6290, 0.6506, 0.6747, 0.6732, 0.6370, 0.6627, 0.6672, 0.6837, 0.6807, 0.6822]
    test_accuracy_dapo = [0.6290, 0.6476, 0.6777, 0.6491, 0.6702, 0.6627, 0.6687, 0.6958, 0.6777, 0.7093]
    test_accuracy_bapo = [0.6290, 0.6898, 0.6792, 0.6943, 0.7018, 0.6988, 0.7139, 0.7093, 0.7410, 0.7272]
    # Create figure with same size
    plt.figure(figsize=(8, 6))
    
    # Same color as original BAPO
    color1 = '#7d3c98'  # Darker Purple
    color2 = '#f39c12' 

    # Plot scatter points and connect with lines
    plt.scatter(steps, test_accuracy_bapo,
                color=color1, alpha=0.7, s=150, marker='o', label='BAPO', zorder=3)
    # Connect with solid line
    plt.plot(steps, test_accuracy_bapo, color=color1, linewidth=5,
              alpha=0.9, linestyle='-', zorder=2)
    
    plt.scatter(steps, test_accuracy_grpo,
                color=color2, alpha=0.7, s=150, marker='^', label='GRPO', zorder=3)

    plt.plot(steps, test_accuracy_grpo, color=color2, linewidth=5,
              alpha=0.9, linestyle='--', zorder=2)
    
    # Styling - exactly same as original
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Test Accuracy (↑)', fontsize=24)
    plt.title('AMC Test Set', fontsize=24, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Grid - same bold lines
    plt.grid(True, alpha=0.6, linestyle='-', linewidth=2)
    
    # Set bold border
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # Legend
    plt.legend(fontsize=24)
    
    # Clean layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig('bapo_test_accuracy.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'bapo_test_accuracy.png'")

if __name__ == "__main__":
    create_bapo_plot()