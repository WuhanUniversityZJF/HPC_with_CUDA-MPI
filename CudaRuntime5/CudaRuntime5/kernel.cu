#include "gdal_priv.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <omp.h>  // 引入OpenMP头文件

#include <cuda_runtime.h>

using namespace std;
//
struct InputCoordinate {
    float x; // 经度
    float y; // 纬度
    float height;
    float radius;
};

struct ViewPoint {
    double x;
    double y;
    float height;
    float radius;
    int cellX;
    int cellY;
};

__device__ float deviceGetLineNetSpeed(
    const ViewPoint& vp, int endRow, int endCol,
    const float* d_buffer, int xSize, int ySize, float xMin, float yMin,
    float leftTopX, float leftTopY, float xResolution, float yResolution)
{
    int startCol = vp.cellX;
    int startRow = vp.cellY;

    float X2 = leftTopX + endCol * xResolution;
    float Y2 = leftTopY + endRow * yResolution;
    float X1 = vp.x;
    float Y1 = vp.y;
    float Z1 = vp.height;

    int localX = (int)(endCol - xMin);
    int localY = (int)(endRow - yMin);
    if (localX < 0 || localX >= xSize || localY < 0 || localY >= ySize) {
        return 1000.0;
    }

    float Z2 = d_buffer[localY * xSize + localX];
    if (Z2 < 0) {
        return 1000.0;
    }

    if (X1 == X2 && Y1 == Y2) {
        return 0.0;
    }

    float imgCellWidth = xResolution;
    float imgCellHeight = fabs(yResolution);

    float l = sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2));

    bool moreAlongX = (fabs((X1 - X2) / imgCellWidth) > fabs((Y1 - Y2) / imgCellHeight));

    float tree_Penetrate_Distance = 0.0;
    int d = 0;
    float divide_Z = 0.0;
    float divide_S = 0.0;

    if (moreAlongX) {
        float k = (Y2 - Y1) / (X2 - X1);
        d = abs(endCol - startCol);
        if (d == 0) d = 1;

        divide_Z = (Z1 - Z2) / d;
        divide_S = l / d;

        int countBlock = 0;
        for (int i = 0; i < d; ++i) {
            int tmpCol, tmpRow;
            float tmpZ;
            if (endCol > startCol) {
                tmpCol = startCol + i;
                tmpRow = (int)floor(startRow - i * k);
                tmpZ = Z1 + i * (-divide_Z);
            }
            else {
                tmpCol = startCol - i;
                tmpRow = (int)floor(startRow + i * k);
                tmpZ = Z1 + i * (-divide_Z);
            }

            int colLocal = tmpCol - (int)xMin;
            int rowLocal = tmpRow - (int)yMin;
            if (colLocal < 0 || colLocal >= xSize || rowLocal < 0 || rowLocal >= ySize) {
                continue;
            }
            float terrainZ = d_buffer[rowLocal * xSize + colLocal];
            if (tmpZ < terrainZ) {
                countBlock++;
            }
        }
        tree_Penetrate_Distance = countBlock * divide_S;
    }
    else {
        float k = (X2 - X1) / (Y2 - Y1);
        d = abs(endRow - startRow);
        if (d == 0) d = 1;

        divide_Z = (Z1 - Z2) / d;
        divide_S = l / d;

        int countBlock = 0;
        for (int i = 0; i < d; ++i) {
            int tmpCol, tmpRow;
            float tmpZ;
            if (endRow > startRow) {
                tmpRow = startRow + i;
                tmpCol = (int)floor(startCol - i * k);
                tmpZ = Z1 + i * (-divide_Z);
            }
            else {
                tmpRow = startRow - i;
                tmpCol = (int)floor(startCol + i * k);
                tmpZ = Z1 + i * (-divide_Z);
            }

            int colLocal = tmpCol - (int)xMin;
            int rowLocal = tmpRow - (int)yMin;
            if (colLocal < 0 || colLocal >= xSize || rowLocal < 0 || rowLocal >= ySize) {
                continue;
            }
            float terrainZ = d_buffer[rowLocal * xSize + colLocal];
            if (tmpZ < terrainZ) {
                countBlock++;
            }
        }
        tree_Penetrate_Distance = countBlock * divide_S;
    }

    return tree_Penetrate_Distance;
}


__global__ void computeResultsKernel(
    ViewPoint vp, float* d_buffer, float* d_results,
    int xSize, int ySize, int xMin, int yMin,
    float leftTopX, float leftTopY, float xResolution, float yResolution,
    float radius, float cmax, float gamma)
{
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (xx < xSize && yy < ySize) {
        float centerX = xSize / 2.0;
        float centerY = ySize / 2.0;

        float dx = xx - centerX;
        float dy = yy - centerY;
        if ((dx * dx + dy * dy) > radius * radius) {
            d_results[yy * xSize + xx] = -1.0;
        }
        else {
            // 调用deviceGetLineNetSpeed
            float penetrationDistance = deviceGetLineNetSpeed(
                vp, yMin + yy, xMin + xx, d_buffer, xSize, ySize, (float)xMin, (float)yMin,
                leftTopX, leftTopY, xResolution, yResolution
            );
            float grayValue = fmax(fmin(cmax - gamma * penetrationDistance, cmax), (float)0.0);
            d_results[yy * xSize + xx] = grayValue;
        }
    }
}
//
struct Deploy
{
    int width;
    int height;
    double geoTransform[6];
    float leftTopX;
    float leftTopY;
    float xResolution;
    float yResolution;
    GDALDataset* dataset = nullptr;
    std::vector<ViewPoint> viewPoints;

    Deploy(const char* filePath)
    {
        dataset = static_cast<GDALDataset*>(GDALOpen(filePath, GA_ReadOnly));
        if (dataset == nullptr) {
            std::cerr << "Failed to open dataset!" << std::endl;
            std::abort();
        }
        width = dataset->GetRasterXSize();
        height = dataset->GetRasterYSize();
        dataset->GetGeoTransform(this->geoTransform);
        leftTopX = geoTransform[0];
        leftTopY = geoTransform[3];
        xResolution = geoTransform[1];
        yResolution = geoTransform[5];
    }

    void setObserver(const std::vector<InputCoordinate>& in) {
        OGRSpatialReference source, target;
        source.importFromEPSG(4326);
        target.importFromWkt(dataset->GetProjectionRef());
        auto ct = OGRCreateCoordinateTransformation(&source, &target);
        viewPoints.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& ic = in[i];
            ViewPoint& v = viewPoints[i];
            v = { ic.x, ic.y, ic.height, ic.radius };
            ct->Transform(1, &(v.y), &(v.x), nullptr);
            v.cellX = static_cast<int>((v.x - leftTopX) / xResolution);
            v.cellY = static_cast<int>((v.y - leftTopY) / yResolution);
            v.cellX = std::min(std::max(v.cellX, 0), width - 1);
            v.cellY = std::min(std::max(v.cellY, 0), height - 1);
        }
    }

    // 将原逻辑迁移至 device 函数

    // CUDA kernel 函数：并行计算结果


    void handle() {
        float cmax = 40.0f;
        float gamma = 0.5f; // Example gamma value

        // 定义若干组待测试的block size配置
        std::vector<std::pair<int, int>> testBlockConfigs = {
            {32,32}
        };

#pragma omp parallel for
        for (int vpIndex = 0; vpIndex < (int)viewPoints.size(); vpIndex++) {
            const auto& vp = viewPoints[vpIndex];
            int xMin = std::max(0,
                static_cast<int>(std::round(vp.cellX - vp.radius / xResolution)));
            int xMax = std::min(width - 1,
                static_cast<int>(std::round(vp.cellX + vp.radius / xResolution)));
            int yMin = std::max(0,
                static_cast<int>(std::round(vp.cellY - vp.radius / fabs(yResolution))));
            int yMax = std::min(height - 1,
                static_cast<int>(std::round(vp.cellY + vp.radius / fabs(yResolution))));

            int xSize = xMax - xMin + 1;
            int ySize = yMax - yMin + 1;
            std::vector<float> buffer(xSize * ySize);

            CPLErr err = dataset->GetRasterBand(1)->RasterIO(
                GF_Read, xMin, yMin, xSize, ySize, buffer.data(), xSize, ySize, GDT_Float32, 0, 0);
            if (err != CE_None) {
                std::cerr << "Error reading raster data!" << std::endl;
                continue;
            }

            // 分配GPU内存并拷贝数据
            float* d_buffer = nullptr;
            float* d_results = nullptr;
            cudaMalloc((void**)&d_buffer, xSize * ySize * sizeof(float));
            cudaMalloc((void**)&d_results, xSize * ySize * sizeof(float));
            cudaMemcpy(d_buffer, buffer.data(), xSize * ySize * sizeof(float), cudaMemcpyHostToDevice);

            // 循环测试不同的block配置
            for (auto& cfg : testBlockConfigs) {
                int blockSizeX = cfg.first;
                int blockSizeY = cfg.second;

                dim3 block(blockSizeX, blockSizeY);
                dim3 grid((xSize + block.x - 1) / block.x, (ySize + block.y - 1) / block.y);

                double start = omp_get_wtime();
                computeResultsKernel << <grid, block >> > (
                    vp, d_buffer, d_results,
                    xSize, ySize, xMin, yMin,
                    leftTopX, leftTopY, xResolution, yResolution,
                    vp.radius, cmax, gamma
                    );
                cudaDeviceSynchronize();
                double end = omp_get_wtime();
                double elapsed = end - start;
                std::cout << "vpIndex:" << vpIndex << ", block(" << blockSizeX << "," << blockSizeY << ") time: " << elapsed << " seconds." << std::endl;
            }

            // 对于最终结果输出仍可选择一种配置的结果拷回CPU做输出 
            // 这里以最后一次配置结果为例：
            std::vector<float> results(xSize * ySize);
            cudaMemcpy(results.data(), d_results, xSize * ySize * sizeof(float), cudaMemcpyDeviceToHost);

            // 释放GPU内存
            cudaFree(d_buffer);
            cudaFree(d_results);

            std::string filename = "output_" + std::to_string(vpIndex) + ".asc";
            std::ofstream outFile(filename);
            if (!outFile) {
                std::cerr << "Failed to open output file!" << std::endl;
                continue;
            }

            outFile << "NCOLS " << xSize << "\n";
            outFile << "NROWS " << ySize << "\n";
            outFile << "XLLCORNER " << leftTopX + xMin * xResolution << "\n";
            outFile << "YLLCORNER " << leftTopY + yMax * yResolution << "\n";
            outFile << "CELLSIZE " << xResolution << "\n";
            outFile << "NODATA_VALUE -1\n";

            for (int yy = 0; yy < ySize; ++yy) {
                for (int xx = 0; xx < xSize; ++xx) {
                    outFile << results[yy * xSize + xx] << " ";
                }
                outFile << "\n";
            }

            outFile.close();
        }
    }

    ~Deploy() {
        GDALClose(dataset);
    }
};


int main() {
    const char* filePath = "G:\\wuhan\\wuhan.tif";

    const char* path[] = { "D:\\Shared", "..", nullptr };
    OSRSetPROJSearchPaths(path);

    GDALAllRegister();
    Deploy deploy(filePath);

    std::vector<InputCoordinate> in = {
        {114.368264, 30.533097, 68,250},
        {114.368264, 30.533097, 68,500},
        {114.368264, 30.533097, 68,1000},
        {114.368264, 30.533097, 68,2000},
        // 在此添加更多坐标点来测试并行效果
    };
    deploy.setObserver(in);
    deploy.handle();

    return 0;
}























