#ifndef DBLMDB_H
#define DBLMDB_H

#include "lmdb.h"
#include "glog/logging.h"

#include "basic/common.h"

namespace surfing
{
	using namespace std;
	enum Mode { READ, WRITE, NEW };

	inline void MDB_CHECK(int mdb_status)
	{
		CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
	}

	class LMDBTransaction
	{
	public:
		explicit LMDBTransaction(MDB_env* mdb_env) : mdb_env_(mdb_env) {}
		void Put(const string& key, const string& value);
		void Commit();

	private:
		MDB_env* mdb_env_;
		vector<string> keys, values;

		void DoubleMapSize();

		DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
	};

	class LMDB
	{
	public:
		LMDB() :mdb_env_(NULL) {}
		~LMDB() { Close(); }

		void Open(const string& source, Mode mode);
		void Close()
		{
			if (mdb_env_ != NULL)
			{
				mdb_dbi_close(mdb_env_, mdb_dbi_);
				mdb_env_close(mdb_env_);
				mdb_env_ = NULL;
			}
		}
		LMDBTransaction* NewTransaction();

		vector<string>& GetData(vector<string>& keys);

	private:
		MDB_env* mdb_env_;
		MDB_dbi mdb_dbi_;
		MDB_val mdb_key_, mdb_data_;
		vector<string> data_;
	};
}

#endif