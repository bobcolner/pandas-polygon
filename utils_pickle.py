from pickle import dump, load


def pickle_dump(object: object, file_name: str):
	with open(file_name, 'wb') as fio:
		dump(object, fio)


def pickle_load(file_name: str) -> object:
	with open(file_name, 'rb') as fio:
		output = load(fio)

	return output
