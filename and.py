from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)

    X, y = prepare_data(df_AND)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()


    model.save(filename=modelName, model_dir="model")

    save_plot(df_AND, model, filename=plotName)

# entry point 
if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(data=AND, modelName = "AND.model", plotName="AND.png", eta=ETA, epochs=EPOCHS)

