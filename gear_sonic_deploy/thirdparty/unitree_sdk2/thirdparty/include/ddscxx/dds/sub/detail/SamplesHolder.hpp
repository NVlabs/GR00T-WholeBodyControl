/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_SAMPLES_HOLDER_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_SAMPLES_HOLDER_HPP_

/**
 * @file
 */

#include <dds/sub/LoanedSamples.hpp>
#include "org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp"
#include "org/eclipse/cyclonedds/topic/datatopic.hpp"

namespace dds
{
namespace sub
{
namespace detail
{

template <typename T>
class LoanedSamplesHolder : public SamplesHolder
{
public:
    LoanedSamplesHolder(dds::sub::LoanedSamples<T>& samples) : samples_(samples), index_(0)
    {
    }

    void set_length(uint32_t len) {
        this->samples_.delegate()->resize(len);
    }

    uint32_t get_length() const {
        return this->index_;
    }

    SamplesHolder& operator++(int)
    {
        this->index_++;
        return *this;
    }

    void *data()
    {
        return (*this->samples_.delegate())[this->index_].delegate().data_ptr();
    }

    detail::SampleInfo& info()
    {
        return (*this->samples_.delegate())[this->index_].delegate().info();
    }

    void **cpp_sample_pointers(size_t length)
    {

        return new void * [length];
    }

    dds_sample_info_t *cpp_info_pointers(size_t length)
    {
        return new dds_sample_info_t[length];
    }

    void set_sample_contents(void** c_sample_pointers, dds_sample_info_t *info)
    {
        uint32_t cpp_sample_size = this->samples_.delegate()->length();
        auto tmp_iterator = samples_.delegate()->mbegin();
        for (uint32_t i = 0; i < cpp_sample_size; ++i, ++tmp_iterator) {
            /* Transfer ownership for ddscxx_serdata from c_sample_pointers to SampleRef objects. */
            tmp_iterator->delegate().data_ptr() = static_cast<ddscxx_serdata<T>*>(c_sample_pointers[i]);
            org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(info[i], tmp_iterator->delegate().info());
        }
    }

    void fini_samples_buffers(void**& c_sample_pointers, dds_sample_info_t*& c_sample_infos)
    {
        delete [] c_sample_pointers;
        delete [] c_sample_infos;
    }

private:
    dds::sub::LoanedSamples<T>& samples_;
    uint32_t index_;
};

class CDRSamplesHolder : public SamplesHolder
{
public:
    CDRSamplesHolder(dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob>& samples) : samples_(samples), index_(0)
    {
    }

    void set_length(uint32_t len) {
        this->samples_.delegate()->resize(len);
    }

    uint32_t get_length() const {
        return this->index_;
    }

    SamplesHolder& operator++(int)
    {
        this->index_++;
        return *this;
    }

    void *data()
    {
        return (*this->samples_.delegate())[this->index_].delegate().data_ptr();
    }

    detail::SampleInfo& info()
    {
        return (*this->samples_.delegate())[this->index_].delegate().info();
    }

    void **cpp_sample_pointers(size_t length)
    {
        return new void * [length];
    }

    dds_sample_info_t *cpp_info_pointers(size_t length)
    {
        return new dds_sample_info_t[length];
    }

    void set_sample_contents(void** c_sample_pointers, dds_sample_info_t *info)
    {
      struct ddsi_serdata **cdr_blobs = reinterpret_cast<struct ddsi_serdata **>(c_sample_pointers);
      const uint32_t cpp_sample_size = this->samples_.delegate()->length();
      for (uint32_t i = 0; i < cpp_sample_size; ++i)
      {
        struct ddsi_serdata * current_blob = cdr_blobs[i];
        org::eclipse::cyclonedds::topic::CDRBlob &sample_data = (*this->samples_.delegate())[i].delegate().data();
        // update the data kind
        sample_data.kind(static_cast<org::eclipse::cyclonedds::topic::BlobKind>(current_blob->kind));

        // if data is transferred using SHM, update the CDRBlob with iox_chunk
        if(!update_cdrblob_from_iox_chunk(*current_blob, sample_data)) {
          ddsrt_iovec_t blob_content;
          ddsi_serdata_to_ser_ref(current_blob, 0, ddsi_serdata_size(current_blob), &blob_content);
          copy_buffer_to_cdr_blob(reinterpret_cast<uint8_t *>(blob_content.iov_base),
                                  blob_content.iov_len, sample_data.kind(), sample_data);
          ddsi_serdata_to_ser_unref(current_blob, &blob_content);
          ddsi_serdata_unref(current_blob);
        }
        // copy sample infos
        org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(info[i], (*samples_.delegate())[i].delegate().info());
      }
    }

    void fini_samples_buffers(void**& c_sample_pointers, dds_sample_info_t*& c_sample_infos)
    {
        delete [] c_sample_pointers;
        delete [] c_sample_infos;
    }

private:
    dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob>& samples_;
    uint32_t index_;

    void copy_buffer_to_cdr_blob(const uint8_t * buffer, const size_t size,
                                 const org::eclipse::cyclonedds::topic::BlobKind data_kind,
                                 org::eclipse::cyclonedds::topic::CDRBlob & cdr_blob) const
    {
      // update the CDR header
      memcpy(cdr_blob.encoding().data(), buffer, CDR_HEADER_SIZE);
      // if the data kind is not empty
      if (data_kind != org::eclipse::cyclonedds::topic::BlobKind::Empty) {
        // get the actual data from the buffer
        cdr_blob.payload().assign(buffer + CDR_HEADER_SIZE, buffer + size);
      }
    }

  bool update_cdrblob_from_iox_chunk (ddsi_serdata & current_blob,
                                      org::eclipse::cyclonedds::topic::CDRBlob &sample_data) {
#ifdef DDSCXX_HAS_SHM
        // if the data is available on SHM
        if (current_blob.iox_chunk && current_blob.iox_subscriber) {
            // get the user iox header
            auto iox_header = iceoryx_header_from_chunk(current_blob.iox_chunk);
            // if the iox chunk has the data in serialized form
            if (iox_header->shm_data_state == IOX_CHUNK_CONTAINS_SERIALIZED_DATA) {
              copy_buffer_to_cdr_blob(reinterpret_cast<uint8_t *>(current_blob.iox_chunk),
                                      iox_header->data_size, sample_data.kind(), sample_data);
            } else if (iox_header->shm_data_state == IOX_CHUNK_CONTAINS_RAW_DATA) {
              // serialize the data
              auto serialized_size = ddsi_sertype_get_serialized_size(current_blob.type,
                                                                      current_blob.iox_chunk);
              // create a buffer to serialize
              std::vector<uint8_t> buffer(serialized_size);
              // serialize into the buffer
              ddsi_sertype_serialize_into(current_blob.type, current_blob.iox_chunk, buffer.data(),
                                          serialized_size);
              // update the CDR blob with the serialized data
              copy_buffer_to_cdr_blob(buffer.data(), serialized_size, sample_data.kind(), sample_data);
            } else {
              // this shouldn't never happen
              ISOCPP_THROW_EXCEPTION(ISOCPP_PRECONDITION_NOT_MET_ERROR,
                                     "The received sample over SHM is not initialized");
            }
            // release the chunk
            free_iox_chunk(static_cast<iox_sub_t *>(current_blob.iox_subscriber), &current_blob.iox_chunk);
            return true;
        } else {
          return false;
        }
#else
        (void) current_blob;
        (void) sample_data;
        return false;
#endif  // DDSCXX_HAS_SHM
    }
};

template <typename T, typename SamplesFWIterator>
class SamplesFWInteratorHolder : public SamplesHolder
{
public:
    SamplesFWInteratorHolder(SamplesFWIterator& it) : iterator(it), size(0)
    {
    }

    void set_length(uint32_t len) {
        this->size = len;

    }

    uint32_t get_length() const {
        return this->size;
    }

    SamplesHolder& operator++(int)
    {
        ++this->iterator;
        return *this;
    }

    void *data()
    {
        return (*iterator).delegate().data_ptr();
    }

    detail::SampleInfo& info()
    {
        return (*iterator).delegate().info();
    }

    void **cpp_sample_pointers(size_t length)
    {
        void **c_sample_pointers = new void * [length];
        SamplesFWIterator tmp_iterator = iterator;
        for (uint32_t i = 0; i < length; ++i, ++tmp_iterator) {
            c_sample_pointers[i] = (*tmp_iterator).delegate().data_ptr();
        }
        return c_sample_pointers;
    }

    dds_sample_info_t *cpp_info_pointers(size_t length)
    {
      return new dds_sample_info_t[length];
    }

    void set_sample_contents(void**, dds_sample_info_t *info)
    {
        /* Samples have already been deserialized in their containers during the read/take call. */
        SamplesFWIterator tmp_iterator = iterator;
        for (uint32_t i = 0; i < size; ++i, ++tmp_iterator) {
            org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(info[i], tmp_iterator->delegate().info());
        }
    }

    void fini_samples_buffers(void**& c_sample_pointers, dds_sample_info_t*& c_sample_infos)
    {
        delete [] c_sample_infos;
        delete [] c_sample_pointers;
    }

private:
    SamplesFWIterator& iterator;
    uint32_t size;

};

template <typename T, typename SamplesBIIterator>
class SamplesBIIteratorHolder : public SamplesHolder
{
public:
    SamplesBIIteratorHolder(SamplesBIIterator& it) : iterator(it), size(0)
    {
    }

    void set_length(uint32_t len) {
        this->size = len;
        samples.resize(len);
    }

    uint32_t get_length() const {
        return this->size;
    }

    SamplesHolder& operator++(int)
    {
        ++this->iterator;
        return *this;
    }

    void *data()
    {
        return this->samples[0].delegate().data_ptr();
    }

    detail::SampleInfo& info()
    {
        return this->samples[0].delegate().info();
    }

    void **cpp_sample_pointers(size_t length)
    {
        set_length(static_cast<uint32_t>(length));
        void **c_sample_pointers = new void*[length];
        for (uint32_t i = 0; i < length; ++i) {
          c_sample_pointers[i] = samples[i].delegate().data_ptr();
        }
        return c_sample_pointers;
    }

    dds_sample_info_t *cpp_info_pointers(size_t length)
    {
        dds_sample_info_t *c_info_pointers = new dds_sample_info_t[length];
        return c_info_pointers;
    }

    void set_sample_contents(void**, dds_sample_info_t *info)
    {
        /* Samples have already been deserialized in their containers during the read/take call. */
        for (uint32_t i = 0; i < size; ++i) {
            org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(info[i], samples[i].delegate().info());
            this->iterator = std::move(samples[i]);
            this->iterator++;
        }
    }

    void fini_samples_buffers(void**& c_sample_pointers, dds_sample_info_t*& c_sample_infos)
    {
        delete [] c_sample_infos;
        delete [] c_sample_pointers;
    }

private:
    SamplesBIIterator& iterator;
    std::vector< dds::sub::Sample<T, dds::sub::detail::Sample> > samples;
    uint32_t size;

};

}
}
}




#endif /* CYCLONEDDS_DDS_SUB_DETAIL_SAMPLES_HOLDER_HPP_ */
