# Machine Learning Operations Project 

The following pages contain some basic documentation for our project for MLOps course at DTU 2025.
We created a baseline model and used a pretrained model from huggingface (timm) to predict pneunomia for an X-Ray image dataset.



## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        modeltraining.md  # Markdown for our model training and other related functions
        api.md # Markdown page for api 
    dockerfiles/
    experiments/
    models/
    src/chest_xray_diagnosis/
        __pycache__/
        configs/
            sweep.yaml
        logs/
        site/
        __innit__.py
        api.py
        data.py
        evaluate.py
        frontend.py
        model.py
        train.py
        visualize.py
    tests/
        __init__.py
        test_api.py
        test_data.py
        test_model.py
    README.md
    requirements_dev.txt
    requirements_tests.tx
    requirements.txt
    tasks.py


