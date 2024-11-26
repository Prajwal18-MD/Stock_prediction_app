import shap

class ModelInterpretability:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def shap_explanation(self):
        """
        Generate SHAP explanations.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.data)
        shap.summary_plot(shap_values, self.data, show=False)
