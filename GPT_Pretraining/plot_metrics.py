import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import configparser
import os

config = configparser.ConfigParser()
config.read("config.ini")

PLOTS_FOLDER = config["PATHS"]["PLOTS_FOLDER"]

class Plots:

    def __init__(self, num_samples, epochs, train_values, val_values, label = 'loss'):
        
        self.num_samples = num_samples
        self.epochs = epochs
        self.train_values = train_values
        self.val_values = val_values

    def plots(self, label = 'loss', type='pretrain'):
        fig, ax = plt.subplots(figsize=(10,5))

        #Epochs vs Loss
        ax.plot(self.epochs,self.train_values,label = f'Training {label}')
        ax.plot(self.epochs, self.val_values, linestyle="--", label = f'Validation {label}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'{label}')
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        #Token vs Loss
        ax2 = ax.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(self.num_samples, self.train_values)
        ax2.set_xlabel("Samples seen")

        fig.tight_layout()  # Adjust layout to make room
        plot_save_path = os.path.join(PLOTS_FOLDER,f"{label}_samples_epochs_{type}_{str(datetime.now().timestamp())}.pdf")
        plt.savefig(plot_save_path)
        plt.show()

