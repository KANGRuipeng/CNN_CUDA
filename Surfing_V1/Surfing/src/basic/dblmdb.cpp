#include "basic/dblmdb.h"

#include <direct.h>
#define mkdir(X,Y) _mkdir(X)

namespace surfing
{
	void LMDBTransaction::Put(const string& key, const string& value)
	{
		keys.push_back(key);
		values.push_back(value);
	}

	void LMDBTransaction::Commit()
	{
		MDB_dbi mdb_dbi;
		MDB_val mdb_key, mdb_data;
		MDB_txn* mdb_txn;

		MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
		MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

		bool out_of_memory = false;
		for (int i = 0; i < keys.size(); i++)
		{
			mdb_key.mv_size = keys[i].size();
			mdb_key.mv_data = const_cast<char*>(keys[i].data());

			mdb_data.mv_size = values[i].size();
			mdb_data.mv_data = const_cast<char*>(values[i].data());

			int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
			if (put_rc == MDB_MAP_FULL)
			{
				out_of_memory = true;
				break;
			}
			else
			{
				MDB_CHECK(put_rc);
			}
		}
		if (!out_of_memory)
		{
			MDB_CHECK(mdb_txn_commit(mdb_txn));
			mdb_dbi_close(mdb_env_, mdb_dbi);
			keys.clear();
			values.clear();
		}
		else
		{
			mdb_txn_abort(mdb_txn);
			mdb_dbi_close(mdb_env_, mdb_dbi);
			DoubleMapSize();
			Commit();
		}
	}

	void LMDBTransaction::DoubleMapSize()
	{
		struct MDB_envinfo current_info;
		MDB_CHECK(mdb_env_info(mdb_env_, &current_info));
		size_t new_size = current_info.me_mapsize * 2;
		DLOG(INFO) << "Doubling LMDB map size to" << (new_size >> 20) << "MB...";
		MDB_CHECK(mdb_env_set_mapsize(mdb_env_, new_size));
	}

	void LMDB::Open(const string& source, Mode mode)
	{
		MDB_CHECK(mdb_env_create(&mdb_env_));
		if (mode == NEW)
		{
			CHECK_EQ(mkdir(source.c_str(), 0744), 0) << " mkdir " << source << "failed";
		}
		int flags = 0;

		if (mode == READ)
		{
			flags = MDB_RDONLY | MDB_NOTLS;
		}
		int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0644);
		MDB_CHECK(rc);
		DLOG(INFO) << " Open db LMDB " << source;
	}

	vector<string>& LMDB::GetData(vector<string>& keys)
	{
		MDB_txn* mdb_txn;
		MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
		MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
		data_.resize(keys.size());
		for (int i = 0; i < keys.size(); i++)
		{
			mdb_key_.mv_size = keys[i].size();
			mdb_key_.mv_data = const_cast<char*>(keys[i].data());
			MDB_CHECK(mdb_get(mdb_txn, mdb_dbi_, &mdb_key_, &mdb_data_));
			data_[i] = string(static_cast<const char*>(mdb_data_.mv_data), mdb_data_.mv_size);
		}
		mdb_txn_abort(mdb_txn);
		return data_;
	}

	LMDBTransaction* LMDB::NewTransaction()
	{
		return new LMDBTransaction(mdb_env_);
	}

}