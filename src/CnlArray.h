/**
   Copyright 2018 HIGASHIMURA Takenori

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef CNLARRAY_H_
#define CNLARRAY_H_

#include <cstring>

namespace cnl {

using namespace std;

struct Exception {
	std::string message;
	Exception(const char* message) {
		this->message = message;
	}
	Exception(std::string message) {
		this->message = message;
	}
	virtual ~Exception(){}
};

class NumpyObject {
protected:
	const char* header1{"\x93NUMPY"};
	const char* dtype{"<f8"};
	const int PROPSIZE = 128;
	string path;
	FILE*	fp{NULL};
public:
	NumpyObject(){}
	virtual ~NumpyObject(){
		close();
	}
	void writeHeader() {
		char properties[PROPSIZE];
		snprintf(
			properties,
			PROPSIZE,
			"{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
			dtype,
			(const char*)getShape().c_str()
		);
		unsigned short len = strlen(properties);
		unsigned short hsize = ((len + 10) / 16) * 16 + 6;
		writeString((char*)header1);
		writeShort(1);
		writeShort(hsize);
		writeString(properties);
		writeString(string(hsize - len - 1, ' ') + '\n');
	}
	virtual string getShape() = 0;
	void open(string base) {
		path = base;
		fp = fopen(path.c_str(), "wb");
		if (fp == NULL) {
			throw Exception(path + " not opened");
		}
	}
	inline void writeString(const char* p, size_t n=0) {
		if (fp != NULL) {
			if (n == 0) {
				n = strlen(p);
			}
			fwrite(p, 1, n, fp);
		}
	}
	inline void writeString(string s) {
		writeString(s.c_str(), s.size());
	}
	inline void writeShort(short value) {
		if (fp != NULL) {
			fwrite((const char*)&value, sizeof(short), 1, fp);
		}
	}
	inline void writeShort(short* p, size_t n=1) {
		if (fp != NULL) {
			fwrite((const char*)p, sizeof(short), n, fp);
		}
	}
	inline void writeInt(int value) {
		if (fp != NULL) {
			fwrite((const char*)&value, sizeof(int), 1, fp);
		}
	}
	inline void writeInt(int* p, size_t n=1) {
		if (fp != NULL) {
			fwrite((const char*)p, sizeof(int), n, fp);
		}
	}
	inline void writeDouble(double* p, size_t n=1) {
		if (fp != NULL) {
			fwrite((const char*)p, sizeof(double), n, fp);
		}
	}
	inline void close() {
		if (fp != NULL) {
			fclose(fp);
			fp = NULL;
		}
	}
	virtual void save() = 0;
};

class CnlArray {
protected:
	size_t size_x{0};
	size_t length{0};
	double* ptr{nullptr};
public:
	CnlArray(){}
	CnlArray(size_t sx) {
		setExtension(sx);
	}
	virtual ~CnlArray() {
		clearMemory();
	}
	inline void setExtension(size_t sx) {
		size_x = sx;
		length = sx;
		clearMemory();
		ptr = new double[length];
	}
	inline int xlength() {
		return size_x;
	}
	inline size_t getLength() {
		return length;
	}
	inline double get(size_t x) {
		return *(ptr+x);
	}
	inline void set(size_t x, double v) {
		*(ptr+x) = v;
	}
	inline void setAll(double* p){
		memcpy(ptr, p, sizeof(double)*length);
	}
	inline double* head() {
		return ptr;
	}
	inline void clearMemory() {
		if (ptr != nullptr) {
			delete []ptr;
			ptr = nullptr;
		}
	}
	inline void dump() {
		for(size_t i=0; i<size_x; i++) {
			std::cout << i << '\t' << get(i) << endl;
		}
	}
};

class NumpyArray : public NumpyObject, public CnlArray {
public:
	NumpyArray(){}
	NumpyArray(size_t sx) : CnlArray(sx) {
	}
	virtual ~NumpyArray() {
	}
	inline string getShape() {
		char sbuf[32];
		sprintf(sbuf, "(%d, )", this->size_x);
		return string(sbuf);
	}
	inline void save() {
		writeHeader();
		writeDouble(head(), getLength());
	}
};

}

#endif /* CNLARRAY_H_ */
