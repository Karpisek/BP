PYTHON=project_venv/bin/python3

prepare:
	python3 -m pip install virtualenv
	virtualenv project_venv

	source project_venv/bin/activate; \
	project_venv/bin/pip install -Ur requirements.txt; \

run:
	. project_venv/bin/activate
	${PYTHON} detect_violations.py --input=${INPUT} --output=${OUTPUT} -${PARAMS}

install:
	source ./project_venv/bin/activate; \
	pip install -r requirements.txt; \

doc:
	source project_venv/bin/activate; \
	sphinx-build -b html source doc; \
