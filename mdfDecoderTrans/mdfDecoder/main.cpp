#include "MpfDecoder.h"

#include <stdio.h>
#include <io.h>

#define MPF_FILE_PATH "D:\\420.mpf"
#define SAVE_FILE_PATH "D:\\OLHWDB\\OLHWDB1.1trn\\1204_c.txt"
#define SAVE_DIR "D:\\OLHWDB\\OLCS1.0trn\\"

void TransTrainSet(string train_set_dir = "D:\\OLHWDB\\OLHWDB1.0trn\\", string type = "*.mpf") {
	_finddata_t fileDir;
	long lfDir;
	MpfDecoder mpf;

	if ((lfDir = _findfirst((train_set_dir + type).c_str(), &fileDir)) == -1l)
		printf("No file is found\n");
	else {
		printf("file list:\n");
		do {
			printf("%s%s\n", train_set_dir.c_str(), fileDir.name);
			mpf.loadFromFile((train_set_dir + string(fileDir.name)).c_str());
			for (int i = 0; i < mpf.sample_num; ++i) {
				mpf.write_value(mpf.values[i], mpf.labels[i], SAVE_DIR);
			}
		} while (_findnext(lfDir, &fileDir) == 0);
	}
	_findclose(lfDir);
}

void DeleteAll(string train_set_dir = "D:\\OLHWDB\\OLCS1.0trn\\", string type = "*.txt") {
	_finddata_t fileDir;
	long lfDir;
	MpfDecoder mpf;

	if ((lfDir = _findfirst((train_set_dir + type).c_str(), &fileDir)) == -1l)
		printf("No file is found\n");
	else {
		printf("file list:\n");
		do {
			printf("%s%s\n", train_set_dir.c_str(), fileDir.name);

			system(("del " + train_set_dir + string(fileDir.name)).c_str());
		} while (_findnext(lfDir, &fileDir) == 0);
	}
	_findclose(lfDir);
}

void main() {
	DeleteAll();
	TransTrainSet();

	system("pause");
}