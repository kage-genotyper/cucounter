.PHONY: all install uninstall clean

all: install

install: clean
	pip install .

uninstall: clean
	pip uninstall cucounter

clean:
	$(RM) -rf build cucounter.egg-info
