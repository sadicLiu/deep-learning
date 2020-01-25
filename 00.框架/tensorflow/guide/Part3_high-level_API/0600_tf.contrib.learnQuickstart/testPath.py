import os.path as op

if __name__ == '__main__':
	if op.exists("test.txt"):
		print("exists")
	else:
		print("not exists")