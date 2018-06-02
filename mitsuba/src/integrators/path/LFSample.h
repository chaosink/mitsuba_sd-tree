#ifndef LF_SAMPLE_H
#define LF_SAMPLE_H

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/fstream.h>

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <armadillo>

#define POLY_RECORD 0


using namespace mitsuba;

namespace NNA {

	enum LFType
	{
		LF_UNKNOWN = 0,
		LF_PRIMARY_SAMPLE_SAPCE = 1,
		LF_SOLID_ANGLE = 2
	};

	struct LFSample {

		enum {
			DIMENSION_RESOLUTION = 64,
			LF_SIZE = DIMENSION_RESOLUTION * DIMENSION_RESOLUTION
		};

#if POLY_RECORD == 0

		half in_radiance[LF_SIZE][3];
		Frame pix_frame;
		//Spectrum out_radiance[LF_SIZE];
		unsigned short n_hit[LF_SIZE];
		unsigned short n_firsthit = 0;

		LFSample() {
			memset(in_radiance, 0, sizeof(half) * 3 * LF_SIZE);
			memset(n_hit, 0, sizeof(unsigned short) * LF_SIZE);
		}

		inline void serialize(Stream *stream) {
			pix_frame.serialize(stream);
			stream->writeHalfArray((half *)in_radiance, LF_SIZE * 3);
			stream->writeUShortArray(n_hit, LF_SIZE);

		}

		inline void ReadStream(Stream *stream) {
			pix_frame = Frame(stream);
			stream->readHalfArray((half *)in_radiance, LF_SIZE * 3);
			stream->readUShortArray(n_hit, LF_SIZE);
		}

		LFSample(Stream *stream) {
			ReadStream(stream);
		}

		inline void Push(Point2 sample, Spectrum _out_radiance, Spectrum _in_radiance) {

			for (int c = 0; c < 3; c++) {
				// _in_radiance[c] = std::log2f(1.0F + _in_radiance[c]);
				_in_radiance[c] = _in_radiance[c];
			}

			if (_in_radiance.isNaN() || !_in_radiance.isValid()) {
				return;
			}

			const float stride = 1.0f / DIMENSION_RESOLUTION;
			uint32_t idx_x = std::min((int)(sample.x / stride), DIMENSION_RESOLUTION - 1);
			uint32_t idx_y = std::min((int)(sample.y / stride), DIMENSION_RESOLUTION - 1);

			uint32_t idx = idx_y * DIMENSION_RESOLUTION + idx_x;
			for (int c = 0; c < 3; c++) {
				in_radiance[idx][c] += _in_radiance[c];
			}

			//out_radiance[idx] += _out_radiance;
			n_hit[idx]++;
		}
#else

		struct Curve {

			half center[3][2];     // coordinate center for rgb channels
			half curve_parameter[3][6]; // a*x^2 + b*y^2 + c*x*y + d*x + e*y + f for rgb channels

			std::vector<Point2> temporal_coordinate;
			std::vector<Spectrum> temporal_in_radiance;


			Curve() {

				for (int i = 0; i < 3; i++) {
					curve_parameter[i][0] = curve_parameter[i][1] = curve_parameter[i][2] = curve_parameter[i][3] = curve_parameter[i][4] = curve_parameter[i][5] = curve_parameter[i][6] = 0.0f;
					center[i][0] = center[i][1] = center[i][2] = 0.0f;
				}
			}

			inline void serialize(Stream *stream) {
				for (int i = 0; i < 3; i++) {
					stream->writeHalfArray<2>(center[i]);
					stream->writeHalfArray<6>(curve_parameter[i]);
				}
			}

			Curve(Stream *stream) {
				for (int i = 0; i < 3; i++) {
					stream->readHalfArray<2>(center[i]);
					stream->readHalfArray<6>(curve_parameter[i]);
				}
			}



			void compute() {

				if (temporal_coordinate.empty()) return;

				for (int c = 0; c < 3; c++) {

					int n_row = temporal_coordinate.size();

					int max_idx = -1;
					float max_value = -0.1f;
					for (int i = 0; i < n_row; i++) {
						float v = temporal_in_radiance[i][c];
						if (v > max_value) {
							max_value = v;
							max_idx = i;
						}
					}

					Point2 this_center = temporal_coordinate[max_idx];

					bool success = false;

					// debug
					if (n_row >= 6) {

						arma::fvec x(n_row);
						arma::fvec y(n_row);
						arma::fvec z(n_row);
						for (int i = 0; i < n_row; i++) {
							x[i] = temporal_coordinate[i].x - this_center.x;
							y[i] = temporal_coordinate[i].y - this_center.y;
							//z[i] = (temporal_in_radiance[i][c]) / temporal_in_radiance.size();
							z[i] = (temporal_in_radiance[i][c]);
						}


						arma::fmat A(n_row, 6);
						A.col(0) = x % x;
						A.col(1) = y % y;
						A.col(2) = x % y;
						A.col(3) = x;
						A.col(4) = y;
						A.col(5).ones();
						arma::fvec curve;

						center[c][0] = this_center[0];
						center[c][1] = this_center[1];

						success = arma::solve(curve, A, z, arma::solve_opts::no_approx);

						if (success) {

							for (int p = 0; p < 6; p++) {
								half &set = curve_parameter[c][p];
								set = curve[p];
								if (set.isInfinity() || set.isNan()) {
									success = false;
									break;
								}
							}
						}

						if (!success) {

							A = A.cols(3, 5);
							success = arma::solve(curve, A, z, arma::solve_opts::no_approx);

							if (success) {

								for (int p = 0; p < 3; p++) {
									half &set = curve_parameter[c][p + 3];
									set = curve[p];
									if (set.isInfinity() || set.isNan()) {
										success = false;
										break;
									}

									curve_parameter[c][p] = 0.0f;
								}

							}

						}
					}
					//success = false;
					if (!success) {

						for (int p = 0; p < 5; p++) {
							curve_parameter[c][p] = 0.0f;
						}

						float sum_raidance = 0.0f;
						for (auto &r : temporal_in_radiance) {
							sum_raidance += r[c];
						}

						if (n_row > 0) sum_raidance /= n_row;

						curve_parameter[c][5] = sum_raidance;

					}

				}

				temporal_coordinate.clear();
				temporal_in_radiance.clear();
			}
		};


		Curve lf_curve[LF_SIZE];
		Frame pix_frame;
		Vector pix_avg_normal{ 0.0f };


		LFSample() {

		}

		inline void serialize(Stream *stream) {
			pix_frame.serialize(stream);
			for (int i = 0; i < LF_SIZE; i++) {
				lf_curve[i].serialize(stream);
			}
		}

		inline void ReadStream(Stream *stream) {
			pix_frame = Frame(stream);
			for (int i = 0; i < LF_SIZE; i++) {
				lf_curve[i] = Curve(stream);
			}

		}

		LFSample(Stream *stream) {
			ReadStream(stream);
		}

		inline void Push(Point2 sample, Spectrum _out_radiance, Spectrum _in_radiance) {


			const float stride = 1.0f / DIMENSION_RESOLUTION;
			uint32_t idx_x = std::min((int)(sample.x / stride), DIMENSION_RESOLUTION - 1);
			uint32_t idx_y = std::min((int)(sample.y / stride), DIMENSION_RESOLUTION - 1);

			uint32_t idx = idx_y * DIMENSION_RESOLUTION + idx_x;


			lf_curve[idx].temporal_coordinate.push_back(sample);
			lf_curve[idx].temporal_in_radiance.push_back(_in_radiance);

		}

		inline void compute() {
			for (int i = 0; i < LF_SIZE; i++) {
				lf_curve[i].compute();

				std::vector<Point2>().swap(lf_curve[i].temporal_coordinate);
				std::vector<Spectrum>().swap(lf_curve[i].temporal_in_radiance);

			}
		}

#endif

		inline uint32_t GetID(uint32_t block_idx_x, uint32_t block_idx_y) {
			return block_idx_y * DIMENSION_RESOLUTION + block_idx_x;
		}

		void WriteEXR(std::string fileName) {
			Vector2i size = Vector2i(NNA::LFSample::DIMENSION_RESOLUTION);
			ref<Bitmap> image = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, size);
			for(int y = 0; y < size.y; y++)
				for(int x = 0; x < size.x; x++) {
					Spectrum s;
					int n = n_hit[y * size.x + x];
					s[0] = in_radiance[y * size.x + x][0] / n;
					s[1] = in_radiance[y * size.x + x][1] / n;
					s[2] = in_radiance[y * size.x + x][2] / n;
					image->setPixel(Point2i(x, y), s);
				}
			image->write(Bitmap::EOpenEXR, fileName);
		}
	};

	struct LFSampleRecord {

		LFType lf_type{ LF_SOLID_ANGLE };
		uint32_t n_pixels{ 0 };
		std::vector<LFSample> v_samples;

		uint32_t width = 0;
		uint32_t height = 0;

		inline LFSample &operator[](uint32_t idx) {
			return v_samples[idx];
		}
		inline void Allocate(uint32_t _n_pixels) {
			n_pixels = _n_pixels;
			v_samples.clear();
			v_samples.resize(n_pixels);
		}

		inline void Allocate(uint32_t w, uint32_t h) {
			width = w;
			height = h;
			n_pixels = w * h;
			v_samples.clear();
			v_samples.resize(n_pixels);
		}


		inline void serialize(Stream *stream) {

			stream->writeUInt((uint32_t)lf_type);
			stream->writeUInt(n_pixels);
			stream->writeUInt(width);
			stream->writeUInt(height);

			for (int i = 0; i < n_pixels; i++) {
				v_samples[i].serialize(stream);
			}

		}

		LFSampleRecord() {

		}

		LFSampleRecord(Stream *stream) {
			lf_type = (LFType)stream->readUInt();
			n_pixels = stream->readUInt();
			width = stream->readUInt();
			height = stream->readUInt();

			Allocate(width, height);

			//Allocate(n_pixels);
			for (int i = 0; i < n_pixels; i++) {
				v_samples[i].ReadStream(stream);
			}
		}

		LFSampleRecord(std::string file_name) {

			ref<Stream> file_stream = new FileStream(fs::path(file_name));
			new (this) LFSampleRecord(file_stream);

		}

		inline void WriteToFile(std::string file_name) {

			ref<Stream> file_stream = new FileStream(fs::path(file_name), FileStream::ETruncWrite);
			serialize(file_stream);
			file_stream->flush();

			//int width = 100, height = 100;

			//ref<Bitmap> map = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(width, height), 3);

			//
			//for (int j = 0; j < height; j++) {

			//	for (int i = 0; i < width; i++) {
			//		Spectrum sum(0.0f);
			//		for (int r_idx = 30; r_idx < 31; r_idx++) {
			//			Spectrum v;//
			//			for (int c = 0; c < 3; c++) {
			//				v[c] = v_samples[i + j * width].lf_curve[r_idx].center[c][1];
			//			}
			//
			//			sum += v;
			//		}

			//		map->setPixel(Point2i(i, j), sum);
			//	}
			//}

			//std::string filename("test.pfm");
			//ref<FileStream> stream = new FileStream(filename, FileStream::ETruncWrite);

			//map->write(Bitmap::EPFM, stream);
		}

		void WriteEXRImageSpace(int block_x, int block_y) {
			int block_idx = block_y * NNA::LFSample::DIMENSION_RESOLUTION + block_x;
			Vector2i size(width, height);
			ref<Bitmap> image = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, size);
			for(int y = 0; y < size.y; y++)
				for(int x = 0; x < size.x; x++) {
					Spectrum s;
					int n = v_samples[y * size.x + x].n_hit[block_idx];
					s[0] = v_samples[y * size.x + x].in_radiance[block_idx][0] / n;
					s[1] = v_samples[y * size.x + x].in_radiance[block_idx][1] / n;
					s[2] = v_samples[y * size.x + x].in_radiance[block_idx][2] / n;
					image->setPixel(Point2i(x, y), s);
				}
			image->write(Bitmap::EOpenEXR,
				"LFSampleRecord_" + std::to_string(block_x) + "-" + std::to_string(block_y) + "_image-space.exr");
		}

		void WriteEXRDirSpace(int pixel_x, int pixel_y) {
			int pixel_idx = pixel_y * width + pixel_x;
			v_samples[pixel_idx].WriteEXR(
				"LFSampleRecord_" + std::to_string(pixel_x) + "-" + std::to_string(pixel_y) + "_dir-space.exr");
		}
	};

	inline float FastArcTan(float x)
	{
		return (M_PI / 4) * x - x * (fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
	}

	inline Point2 LocalDir2Coordinate(const Vector &wo) {

		Point2 pp;

		float at = std::atan2(wo.y, wo.x) / (2 * M_PI);
		//float at = FastArcTan(wo.y / wo.x) / (2 * M_PI);

		pp.x = at >= 0 ? at : 1.0f + at;
		pp.y = (1.0f - wo.z * wo.z);

		pp.x = std::min(std::max(pp.x, 0.0f), 1.0f - 1e-6f);
		pp.y = std::min(std::max(pp.y, 0.0f), 1.0f - 1e-6f);

		return pp;
	};

	inline Vector Coordinate2LocalDir(const Point2 &s) {

		Vector v;
		v.z = std::sqrt(std::max(1.0f - s.y, 0.0f));

		float theta = s.x * 2 * M_PI;
		float y_x = std::tan(theta);

		if (std::isinf(y_x)) {
			v.x = 0.0f;
			v.y = (theta > M_PI) ? -std::sqrt(std::min(1.0f, 1.0f - v.z * v.z)) : std::sqrt(std::min(1.0f, 1.0f - v.z * v.z));
		}
		else {
			y_x = std::abs(y_x);

			float abs_x = std::sqrt(std::max(0.0f, (1.0f - v.z * v.z) / (1.0f + (y_x * y_x))));
			v.x = (theta > 0.5 * M_PI && theta < 1.5 * M_PI) ? -abs_x : abs_x;
			v.y = (theta > M_PI) ? (-y_x * abs_x) : (y_x * abs_x);
		}

		return (v);
	};

};




#endif // LF_SAMPLE_H
