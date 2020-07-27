.PHONY: env deps

env:
	conda create -y -n bbo-neurips python=3.6 && \
		source activate bbo-neurips && \
		pip install -r environment.txt && \
		pip install -r submissions/metalearn/requirements.txt

clean-env:
	conda remove -n bbo-neurips --all -y
