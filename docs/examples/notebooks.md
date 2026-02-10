
# 📓 Jupyter Notebooks

Explore DriftWatch through interactive examples. Click the badge to open any notebook directly in Google Colab.

<div class="grid cards" markdown>

-   __Drift Detection Tutorial__

    ---

    A complete walkthrough of the core features: detecting drift in numerical and categorical data, and generating reports.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/VincentCotella/DriftWatch/blob/main/examples/notebooks/drift_detection_tutorial.ipynb){ .md-button .md-button--primary }

-   __CLI Tutorial__

    ---

    Learn how to use the `driftwatch` command-line interface to check for drift and generate reports without writing Python scripts.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/VincentCotella/DriftWatch/blob/main/examples/notebooks/cli_tutorial.ipynb){ .md-button .md-button--primary }

-   __Advanced Customization__

    ---

    Deep dive into customizing thresholds, selecting specific detectors, and using the `DriftExplainer` and `DriftVisualizer` for detailed analysis.

    [:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/VincentCotella/DriftWatch/blob/main/examples/notebooks/advanced_customization.ipynb){ .md-button .md-button--primary }

</div>

## Running Locally

To run these notebooks locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/VincentCotella/DriftWatch.git
cd DriftWatch
pip install ".[all]"
jupyter notebook examples/notebooks/
```
