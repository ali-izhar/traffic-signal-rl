import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator


class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi = dpi

        # Set up better formatting for plots with higher resolution
        plt.rcParams.update(
            {
                "font.size": 24,
                "figure.figsize": (20, 11.25),
                "figure.dpi": self._dpi,
                "savefig.dpi": self._dpi,
                "lines.linewidth": 2.5,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )

        # Use a modern style with better colors
        plt.style.use("ggplot")

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # Force integer ticks for episodes

        plt.plot(data, linewidth=2.5)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))

        # Add mean line and annotation
        mean_val = np.mean(data)
        plt.axhline(y=mean_val, color="r", linestyle="--", alpha=0.7)
        plt.annotate(
            f"Mean: {mean_val:.2f}",
            xy=(len(data) * 0.75, mean_val),
            xytext=(len(data) * 0.75, mean_val + 0.05 * (max_val - min_val)),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        # Add annotations for min and max values
        min_idx = np.argmin(data)
        max_idx = np.argmax(data)
        plt.annotate(
            f"Min: {min_val:.2f}",
            xy=(min_idx, min_val),
            xytext=(min_idx - len(data) * 0.1, min_val - 0.05 * (max_val - min_val)),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        plt.annotate(
            f"Max: {max_val:.2f}",
            xy=(max_idx, max_val),
            xytext=(max_idx - len(data) * 0.1, max_val + 0.05 * (max_val - min_val)),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        fig = plt.gcf()
        fig.savefig(
            os.path.join(self._path, "plot_" + filename + ".png"), dpi=self._dpi
        )
        plt.close()

        # Save data to file
        with open(
            os.path.join(self._path, "plot_" + filename + "_data.txt"), "w"
        ) as file:
            for value in data:
                file.write("%s\n" % value)

    def save_comparative_data_and_plot(
        self, data_dict, filename, xlabel, ylabel, title=None
    ):
        """
        Create comparative plots for multiple agents
        data_dict: Dictionary with agent names as keys and data lists as values
        """
        plt.figure(figsize=(22, 12))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Find global min/max for consistent y-axis
        all_data = [val for sublist in data_dict.values() for val in sublist]
        global_min = min(all_data) if all_data else 0
        global_max = max(all_data) if all_data else 0

        # Plot each agent's data with a different color and line style
        for i, (agent_name, data) in enumerate(data_dict.items()):
            # Smooth data for better visibility using moving average
            window_size = min(5, len(data) // 10) if len(data) > 20 else 1
            if window_size > 1:
                smoothed_data = np.convolve(
                    data, np.ones(window_size) / window_size, mode="valid"
                )
                # Plot both raw and smoothed data
                plt.plot(data, alpha=0.3, label=f"{agent_name} (raw)")
                plt.plot(
                    range(window_size - 1, len(data)),
                    smoothed_data,
                    linewidth=3,
                    label=f"{agent_name}",
                )
            else:
                plt.plot(data, linewidth=3, label=agent_name)

            # Add mean line
            if len(data) > 0:
                mean_val = np.mean(data)
                plt.axhline(
                    y=mean_val,
                    color=plt.gca().lines[-1].get_color(),
                    linestyle="--",
                    alpha=0.5,
                )

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        plt.margins(0)

        # Set y-axis limits with padding
        padding = (
            0.05 * (global_max - global_min)
            if global_max != global_min
            else 0.05 * abs(global_min)
        )
        plt.ylim(global_min - padding, global_max + padding)

        # Add title if provided
        if title:
            plt.title(title)

        # Add legend with better positioning
        plt.legend(loc="best", fontsize=16)

        # Save the plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(self._path, "plot_" + filename + ".png"), dpi=self._dpi
        )
        plt.close()

        # Save the comparative data to a file
        with open(
            os.path.join(self._path, "plot_" + filename + "_data.txt"), "w"
        ) as file:
            for agent_name, data in data_dict.items():
                file.write(f"{agent_name}: {','.join(map(str, data))}\n")

    def save_comparative_statistics(self, data_dict, filename):
        """
        Save statistical summary of all agents' performance
        """
        with open(os.path.join(self._path, filename + "_statistics.txt"), "w") as file:
            file.write("Agent Type, Min, Max, Mean, Median, Final Value\n")

            for agent_name, data in data_dict.items():
                if not data:
                    continue

                min_val = min(data)
                max_val = max(data)
                mean_val = sum(data) / len(data)
                median_val = sorted(data)[len(data) // 2]
                final_val = data[-1]

                file.write(
                    f"{agent_name}, {min_val}, {max_val}, {mean_val}, {median_val}, {final_val}\n"
                )

    def save_comparative_bar_chart(self, data_dict, filename, ylabel, title=None):
        """
        Create a bar chart to compare performance metrics across agents
        data_dict: Dictionary with agent names as keys and metric values as values
        """
        plt.figure(figsize=(20, 10))

        # Sort agents by metric value for better visualization
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        agent_names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Create color map for different agent types
        colors = {
            "dqn": "#1f77b4",  # blue
            "qlearning": "#ff7f0e",  # orange
            "a2c": "#2ca02c",  # green
            "ppo": "#d62728",  # red
            "fixed": "#9467bd",  # purple
            "actuated": "#8c564b",  # brown
            "webster": "#e377c2",  # pink
        }

        # Use agent colors or default colors if agent not in predefined list
        bar_colors = [colors.get(agent, "#7f7f7f") for agent in agent_names]

        # Create bars
        bars = plt.bar(agent_names, values, color=bar_colors, width=0.6)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(values),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=18,
            )

        # Add styling
        plt.ylabel(ylabel, fontsize=20)
        plt.grid(axis="y", alpha=0.3)

        # Add title if provided
        if title:
            plt.title(title, fontsize=24)

        # Add category labels
        rl_agents = [
            i
            for i, agent in enumerate(agent_names)
            if agent in ["dqn", "qlearning", "a2c", "ppo"]
        ]
        traditional_agents = [
            i
            for i, agent in enumerate(agent_names)
            if agent in ["fixed", "actuated", "webster"]
        ]

        if rl_agents:
            plt.axvspan(
                min(rl_agents) - 0.5, max(rl_agents) + 0.5, alpha=0.1, color="blue"
            )
            plt.text(
                np.mean(rl_agents),
                max(values) * 1.15,
                "RL Agents",
                ha="center",
                va="center",
                fontsize=16,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        if traditional_agents:
            plt.axvspan(
                min(traditional_agents) - 0.5,
                max(traditional_agents) + 0.5,
                alpha=0.1,
                color="red",
            )
            plt.text(
                np.mean(traditional_agents),
                max(values) * 1.15,
                "Traditional Controllers",
                ha="center",
                va="center",
                fontsize=16,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30, ha="right", fontsize=16)

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(
            os.path.join(self._path, "plot_" + filename + ".png"), dpi=self._dpi
        )
        plt.close()

        # Save the data to a file
        with open(
            os.path.join(self._path, "plot_" + filename + "_data.txt"), "w"
        ) as file:
            for agent_name, value in sorted_items:
                file.write(f"{agent_name}: {value}\n")
