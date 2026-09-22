
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace propagation::raytracing::optix
{
#define CUDA_CHECK(call)                                                                                               \
	{                                                                                                                  \
		cudaError_t rc = call;                                                                                         \
		if (rc != cudaSuccess)                                                                                         \
		{                                                                                                              \
			throw std::runtime_error(std::format("[CUDA Error] {}:{}  {}: {}", __FILE_NAME__, __LINE__,                \
												 cudaGetErrorName(rc), cudaGetErrorString(rc)));                       \
		}                                                                                                              \
	}

#define CUDA_RES_CHECK(call)                                                                                           \
	{                                                                                                                  \
		CUresult res = call;                                                                                           \
		if (res != CUDA_SUCCESS)                                                                                       \
		{                                                                                                              \
			throw std::runtime_error(                                                                                  \
				std::format("[CUDA Error] {}:{} Error code {}", __FILE_NAME__, __LINE__, static_cast<int>(res)));      \
		}                                                                                                              \
	}

#define CUDA_SYNC_CHECK()                                                                                              \
	{                                                                                                                  \
		cudaDeviceSynchronize();                                                                                       \
		cudaError_t error = cudaGetLastError();                                                                        \
		if (error != cudaSuccess)                                                                                      \
		{                                                                                                              \
			throw std::runtime_error(                                                                                  \
				std::format("[CUDA Error] {}:{} Error: {}", __FILE__, __LINE__, cudaGetErrorString(error)));           \
		}                                                                                                              \
	}

#define OPTIX_CHECK(call)                                                                                              \
	{                                                                                                                  \
		OptixResult res = call;                                                                                        \
		if (res != OPTIX_SUCCESS)                                                                                      \
		{                                                                                                              \
			throw std::runtime_error(std::format("[OptiX Error] {}:{} Call to {} failed with code {}", __FILE_NAME__,  \
												 __LINE__, #call, static_cast<int>(res)));                             \
		}                                                                                                              \
	}

#define OPTIX_LOG(log, log_size)                                                                                       \
	{                                                                                                                  \
		if ((log_size) > 1)                                                                                            \
			std::cerr << __FILE_NAME__ << ':' << __LINE__ << " OptiX Log: " << &(log) << '\n';                         \
                                                                                                                       \
		log_size = sizeof((log));                                                                                      \
	}

	/**
	 * @brief Wrapper for creating and managing a device-side CUDA buffer.
	 *
	 * Buffer operations have stream-ordered semantics rather than synchronous semantics.
	 */
	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer(CUstream stream) : _stream(stream) {}
		~DeviceBuffer() { free(); }

		// Delete the copy contructor and copy assignment operator.
		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer& operator=(const DeviceBuffer&) = delete;

		DeviceBuffer(DeviceBuffer&& other) noexcept :
			_stream(other._stream), _ptr(other._ptr), _capacity(other._capacity), _size(other._size)
		{
			other._ptr = 0;
			other._capacity = 0;
			other._size = 0;
		}

		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
		{
			if (this != &other)
			{
				free();
				_stream = other._stream;
				_ptr = other._ptr;
				_capacity = other._capacity;
				_size = other._size;
				other._ptr = 0;
				other._capacity = 0;
				other._size = 0;
			}
			return *this;
		}

		/**
		 * @brief Ensures the buffer can hold at least `n` elements, reallocating only if the
		 * current capacity is insufficient, otherwise the existing allocation (and its contents).
		 * is kept.
		 */
		void reserve(size_t n)
		{
			if (n > _capacity)
			{
				free();
				_capacity = n;
				CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_ptr), n * sizeof(T), _stream));
			}
		}

		/**
		 * @brief Reallocs a larger buffer if necessary, then uploads `v`.
		 *
		 * Only reallocates (growing, never shrinking capacity) if the buffer is too small.
		 */
		void upload(const std::vector<T>& v)
		{
			reserve(v.size());
			_size = v.size();
			CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(_ptr), v.data(), size_bytes(), cudaMemcpyHostToDevice,
									   _stream));
		}
		void upload(const T& d)
		{
			reserve(1);
			_size = 1;
			CUDA_CHECK(
				cudaMemcpyAsync(reinterpret_cast<void*>(_ptr), &d, size_bytes(), cudaMemcpyHostToDevice, _stream));
		}

		void download(std::vector<T>& v)
		{
			assert(_ptr != 0);
			v.resize(size());
			CUDA_CHECK(cudaMemcpyAsync(v.data(), reinterpret_cast<const void*>(_ptr), size_bytes(),
									   cudaMemcpyDeviceToHost, _stream));
		}
		void download(T& d)
		{
			assert(_ptr != 0);
			assert(_size == 1);
			CUDA_CHECK(cudaMemcpyAsync(&d, reinterpret_cast<const void*>(_ptr), size_bytes(), cudaMemcpyDeviceToHost,
									   _stream));
		}

		void free()
		{
			if (_ptr != 0u)
			{
				cudaFreeAsync(reinterpret_cast<void*>(_ptr), _stream);
				_ptr = 0;
				_capacity = 0;
				_size = 0;
			}
		}

		[[nodiscard]] inline CUdeviceptr ptr() const { return _ptr; }
		[[nodiscard]] inline CUdeviceptr offset(uint32_t i) const { return _ptr + i * sizeof(T); }
		[[nodiscard]] inline T* ptr_t() const { return reinterpret_cast<T*>(_ptr); }
		[[nodiscard]] inline size_t size() const { return _size; }
		[[nodiscard]] inline size_t size_bytes() const { return _size * sizeof(T); }
		[[nodiscard]] inline size_t capacity() const { return _capacity; }
		[[nodiscard]] inline size_t capacity_bytes() const { return _capacity * sizeof(T); }

		/**
		 * @brief Override the size of the buffer.
		 *
		 * This is typically useful when the size of the buffer changes due to device-side operations.
		 */
		void set_size(size_t size) { _size = size; }

	private:
		/// The stream operations are performed in.
		CUstream _stream;
		/// CUDA device pointer.
		CUdeviceptr _ptr = 0;
		/// Allocated capacity, by count of T. Only grows (via alloc()/reserve()) until free().
		size_t _capacity = 0;
		/// Logical size in use, by count of T. May be less than _capacity after reserve().
		size_t _size = 0;
	};

	/**
	 * @brief A small helper for using std::pair in `unordered_map`s.
	 */
	struct PairHash
	{
		template <typename T1, typename T2>
		size_t operator()(const std::pair<T1, T2>& p) const
		{
			return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
		}
	};
}
