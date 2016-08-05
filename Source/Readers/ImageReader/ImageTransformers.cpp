//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <algorithm>
#include <unordered_map>
#include <random>
#include "ImageTransformers.h"
#include "Config.h"
#include "ConcStack.h"
#include "StringUtil.h"
#include "ElementTypeUtils.h"
#include <algorithm>

namespace Microsoft { namespace MSR { namespace CNTK
{

struct ImageSequenceData : DenseSequenceData
{
    cv::Mat m_image;
    // In case we do not copy data - we have to preserve the original sequence.
    SequenceDataPtr m_original;
};

void ImageTransformerBase::Initialize(TransformerPtr next,
                                      const ConfigParameters &readerConfig)
{
    Base::Initialize(next, readerConfig);
    m_imageConfig = std::make_unique<ImageConfigHelper>(readerConfig);

    m_seed = readerConfig(L"seed", (unsigned int)0);

    size_t featureStreamId = m_imageConfig->GetFeatureStreamId();
    m_appliedStreamIds.push_back(featureStreamId);
    if (m_appliedStreamIds.size() != 1)
    {
        RuntimeError("Only a single feature stream is supported.");
    }

    const auto &inputStreams = GetInputStreams();
    m_outputStreams.resize(inputStreams.size());
    std::copy(inputStreams.begin(), inputStreams.end(), m_outputStreams.begin());
}

SequenceDataPtr
ImageTransformerBase::Apply(SequenceDataPtr sequence,
                            SequenceDataPtr &sequenceLabel,
                            const StreamDescription &inputStream,
                            const StreamDescription & /*outputStream*/)
{
    assert(inputStream.m_storageType == StorageType::dense);
    auto inputSequence = static_cast<const DenseSequenceData&>(*sequence.get());
    ImageDimensions dimensions(*inputSequence.m_sampleLayout, HWC);
    int columns = static_cast<int>(dimensions.m_width);
    int rows = static_cast<int>(dimensions.m_height);
    int channels = static_cast<int>(dimensions.m_numChannels);

    //cout << "ImageTransformerBase::Apply Colums " << columns << " Rows " << rows << " Channels " << channels << endl;
    //cout << "Sequence " << sequence->m_id << endl;
    

    int typeId = 0;
    if (inputStream.m_elementType == ElementType::tdouble)
    {
        typeId = CV_64F;
    }
    else if (inputStream.m_elementType == ElementType::tfloat)
    {
        typeId = CV_32F;
    }
    else
    {
        RuntimeError("Unsupported type");
    }

    auto result = std::make_shared<ImageSequenceData>();
    int type = CV_MAKETYPE(typeId, channels);


    auto inputSequenceLabel = static_cast<const DenseSequenceData&>(*sequenceLabel.get());
    std::vector<SequenceDataPtr> labelPtr;
    inputSequenceLabel.m_chunk->GetSequence(sequence->m_id, labelPtr);
      
    cv::Mat buffer = cv::Mat(rows, columns, type, inputSequence.m_data);
    Apply(sequence->m_id, buffer, labelPtr[1]);
    if (!buffer.isContinuous())
    {
        buffer = buffer.clone();
    }
    else
    {
        result->m_original = sequence;
    }
    assert(buffer.isContinuous());
    result->m_id = sequence->m_id;
    result->m_image = buffer;
    result->m_data = buffer.ptr();
    result->m_numberOfSamples = inputSequence.m_numberOfSamples;
    
    ImageDimensions outputDimensions(buffer.cols, buffer.rows, buffer.channels());
    result->m_sampleLayout = std::make_shared<TensorShape>(outputDimensions.AsTensorShape(HWC));
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CropTransformer::Initialize(TransformerPtr next,
                                 const ConfigParameters &readerConfig)
{
    ImageTransformerBase::Initialize(next, readerConfig);
    auto featureStreamIds = GetAppliedStreamIds();
    int labelStreamIds = 1;
    cout << "Getinputstreams[0] " << string(GetInputStreams()[featureStreamIds[0]]->m_name.begin(), GetInputStreams()[featureStreamIds[0]]->m_name.end()) << endl;
    cout << "Getinputstreams[1] " << string(GetInputStreams()[labelStreamIds]->m_name.begin(), GetInputStreams()[labelStreamIds]->m_name.end()) << endl;

    /*
    GetAppliedStreamIds() currently only delivers value 0 for featureStreamIds,
    which represents only the features but not the labels.
    For using the labels  GetInputStreams()[1]->m_name can be used.
    */

    //InitFeaturesFromConfig(readerConfig(GetInputStreams()[featureStreamIds[0]]->m_name));
    InitFeaturesFromConfig(readerConfig(GetInputStreams()[featureStreamIds[0]]->m_name));
    InitLabelsFromConfig(readerConfig(GetInputStreams()[labelStreamIds]->m_name));
}

void CropTransformer::InitFeaturesFromConfig(const ConfigParameters &config)
{
    /*
    ImageConfigHelper confhelp(config);
   
    intargvector labelLandmarks = confhelp.GetLabelLandmarks();
    cout << "labelLandmarks " << labelLandmarks[0] << " "<< labelLandmarks.size() << endl;

    size_t labelDim = confhelp.GetLabelDim();
    */

    floatargvector cropRatio = config(L"cropRatio", "1.0");
    m_cropRatioMin = cropRatio[0];
    m_cropRatioMax = cropRatio[1];
    
    if (!(0 < m_cropRatioMin && m_cropRatioMin <= 1.0) ||
        !(0 < m_cropRatioMax && m_cropRatioMax <= 1.0) ||
        m_cropRatioMin > m_cropRatioMax)
    {
        RuntimeError("Invalid cropRatio value, must be > 0 and <= 1. cropMin must "
                     "<= cropMax");
    }

    m_jitterType = ParseJitterType(config(L"jitterType", ""));

    if (!config.ExistsCurrent(L"hflip"))
    {
        m_hFlip = m_imageConfig->GetCropType() == CropType::Random;
    }
    else
    {
        m_hFlip = config(L"hflip");
    }

    m_aspectRatioRadius = config(L"aspectRatioRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
}

void CropTransformer::InitLabelsFromConfig(const ConfigParameters &config)
{
    m_labelDimension = config(L"labelDim");

    if (m_labelDimension<0)
    {
        RuntimeError("Invalid labelDim value, must be > 0 ");
    }

    std::string type = config(L"labelType", "classification");
    
    /*
    Parse parameters if labelType="regression".
    One way of using regression labels is learning landmarks. 
    For Landmarks, parameters like position and visibility are supported. 
    For Visibility it is assumed that the first label corresponds to the first position label, and so on.
    
    */
    if (AreEqualIgnoreCase(type, "regression"))
    {
        if (config.ExistsCurrent("Landmarks"))
        {
            ConfigParameters LandmarkParameters = config("Landmarks");
            
            intargvector LandmarkLabels = LandmarkParameters(L"position_indices", ConfigParameters::Array(intargvector(vector<int>(2,0))));
            m_LandmarkLabels = LandmarkLabels;
            cout << "landmarks " << m_LandmarkLabels.front() << " : " << m_LandmarkLabels.back() << endl;
                      
            intargvector VisibilityLabels = LandmarkParameters(L"visibility_indices", ConfigParameters::Array(intargvector(vector<int>(2, 0))));
            m_VisibilityLabels = VisibilityLabels;
            cout << "visibilities " << m_VisibilityLabels.front() << " : " << m_VisibilityLabels.back() << endl;
            
            std::string relTransformation = LandmarkParameters(L"relative_transformation", "true");
            m_relativeCropping = AreEqualIgnoreCase(relTransformation, "true") ? true : false;
            cout << "relative Crop : " << relTransformation << endl;

            std::string cropLandmark = LandmarkParameters(L"crop_landmark", "soft");
            if (AreEqualIgnoreCase(cropLandmark, "soft"))
            {
                m_cropLandmark = CropModeLandmark::soft;
                cout << "crop_lm = soft" << endl;
            }
            else if (AreEqualIgnoreCase(cropLandmark, "hard"))
            {
                m_cropLandmark = CropModeLandmark::hard;
                cout << "crop_lm = hard" << endl;
            }
            else if (AreEqualIgnoreCase(cropLandmark, "both"))
            {
                m_cropLandmark = CropModeLandmark::both;
                cout << "crop_lm = both" << endl;
            }
            else if (AreEqualIgnoreCase(cropLandmark, "none"))
            {
                m_cropLandmark = CropModeLandmark::none;
                cout << "crop_lm = none" << endl;
            }
            else
            {
                RuntimeError("Invalid value for crop_landmark. Parameter must be either soft/hard/both/none");
            }

            std::string cropVisibility = LandmarkParameters(L"crop_visibility", "hard");
            if (AreEqualIgnoreCase(cropVisibility, "soft"))
            {
                m_cropVisibility = CropModeVisibility::soft;
                cout << "crop_vis = soft" << endl;
            }
            else if (AreEqualIgnoreCase(cropVisibility, "hard"))
            {
                m_cropVisibility = CropModeVisibility::hard;
                cout << "crop_vis = hard" << endl;
            }
            else if (AreEqualIgnoreCase(cropVisibility, "both"))
            {
                m_cropVisibility = CropModeVisibility::both;
                cout << "crop_vis = both" << endl;
            }
            else if (AreEqualIgnoreCase(cropVisibility, "none"))
            {
                m_cropVisibility = CropModeVisibility::none;
                cout << "crop_vis = none" << endl;
            }
            else
            {
                RuntimeError("Invalid value for crop_visibility. Parameter must be either soft/hard/both/none");
            }
            
            m_LandmarkValueMin = LandmarkParameters(L"min_value", 0.0);
            cout << "landmark_min " << m_LandmarkValueMin << endl;

            m_LandmarkValueMax = LandmarkParameters(L"max_value", 2.0);
            cout << "landmark_max " << m_LandmarkValueMax << endl;

            //Check if parameter in configfile are correct
            if ((m_LandmarkLabels.back() == 0) && (m_LandmarkLabels.front() == 0))
            {
                RuntimeError("No Landmark-Labels specified in Section Landmarks");
            }

            if ((m_LandmarkLabels.back() - m_LandmarkLabels.front() + 1) < ((m_VisibilityLabels.back() - m_VisibilityLabels.front() + 1) * 2))
            {
                RuntimeError("Invalid values for ""Landmarks"" and ""Visibilities"". There are more Visibility points than Landmarks ");
            }

            if ((m_LandmarkLabels.back() - m_LandmarkLabels.front() + 1) % 2 != 0)
            {
                RuntimeError("Invalid values for ""Landmarks"". Value range must be an even number, since Landmarks represent 2D Coordinates");
            }

            if ((m_LandmarkLabels.front() > m_LandmarkLabels.back()) ||
                (m_VisibilityLabels.front() > m_VisibilityLabels.back()) ||
                (m_LandmarkLabels.front() < 0) || (m_LandmarkLabels.back() < 0) ||
                (m_VisibilityLabels.front() < 0) || (m_VisibilityLabels.back() < 0))
            {
                RuntimeError("Invalid values at Labels. Notation ""Landmarks"" and ""Visibilities"" must be ranged values. E.g. 1:5.");
            }


            //TODO: What happens if some parameters are omitted , like position and visibility indices
        }

        
    }
}

void CropTransformer::StartEpoch(const EpochConfiguration &config)
{
    cout << endl; cout << endl;
    cout << "CropTransformer:: StartEpoch " << endl;
    m_curAspectRatioRadius = m_aspectRatioRadius[config.m_epochIndex];
    if (!(0 <= m_curAspectRatioRadius && m_curAspectRatioRadius <= 1.0))
        InvalidArgument("aspectRatioRadius must be >= 0.0 and <= 1.0");

    ImageTransformerBase::StartEpoch(config);
    cout << "CropTransformer:: End of StartEpoch " << endl;
}

void CropTransformer::Apply(size_t id, cv::Mat &mat, SequenceDataPtr labelPtr)
{
    cout << "CropTransformer::Apply, Image ID " << id <<endl ;
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    double ratio = 1;
    switch (m_jitterType)
    {
    case RatioJitterType::None:
        ratio = m_cropRatioMin;
        break;
    case RatioJitterType::UniRatio:
        if (m_cropRatioMin == m_cropRatioMax)
        {
            ratio = m_cropRatioMin;
        }
        else
        {
            ratio = UniRealT(m_cropRatioMin, m_cropRatioMax)(*rng);
            assert(m_cropRatioMin <= ratio && ratio < m_cropRatioMax);
        }
        break;
    default:
        RuntimeError("Jitter type currently not implemented.");
    }

    int viewIndex = m_imageConfig->IsMultiViewCrop() ? (int)(id % 10) : 0;

    // TODO: Cut out Label if it gets value outside [0,1]


    cv::Rect cropRect = GetCropRect(m_imageConfig->GetCropType(), viewIndex, mat.rows, mat.cols, ratio, *rng);
    mat = mat(cropRect);

    // LANDMARK CROPPING
    if (m_imageConfig->GetElementType() == ElementType::tfloat)
    {
        float *dat = reinterpret_cast<float*>(labelPtr->m_data);

        //Crop Landmarks
        for (int it_label = 0; it_label < m_labelDimension; it_label = it_label += 2)
        {
            /*
            From Config file, check if selected Label ist soft or hard cropped.
            Landmark labels are regarded as 2D-coordinates (f.e. Landmarks), and will be transformed during Crop-Transformation.
            Also the index of the For Loop should therefore incremented with 2
            Visibility labels are regarded as values and between 0 and 1 and correspont to the Landmarks in sequence.
            If after cropping a Landmark is cut out, the correspondin Visibility label value turns zero         
            */

            /*
            LabelFunction CropMode = LabelFunction::None;

            // Note: Assumption that m_LandmarkLabels and m_VisibilityLabels are vectors with 2 elements
            if ((unsigned)(it_label - m_LandmarkLabels.front()) < (m_LandmarkLabels.back() - m_LandmarkLabels.front()))
            {
                cout << "CropType: Landmark" << endl;
                CropMode = LabelFunction::Landmark;
            }
            else if ((unsigned)(it_label - m_VisibilityLabels.front()) < (m_VisibilityLabels.back() - m_VisibilityLabels.front()))
            {
                cout << "CropType: Visibility" << endl;
                CropMode = LabelFunction::Visibility;
            }
            */

            if (m_cropLandmark == CropModeLandmark::none)
            {
                break;
            }

            // Check if current label is a Landmark
            if ((it_label + 1 < m_LandmarkLabels[0]) || (it_label + 1 >> m_LandmarkLabels[1]))
            {
                continue;
            }

            // Set scaling factor for relative Cropping
            std::vector<float> factor_rel_transform;
            if (m_relativeCropping){
                factor_rel_transform.push_back(cropRect.width);
                factor_rel_transform.push_back(cropRect.height);
            }
            else {
                factor_rel_transform = vector<float>(2, 1);
            }

            float *label_x = &dat[it_label];
            float *label_y = &dat[it_label + 1];
            float test_x = *label_x;
            float test_y = *label_y;
            cout << "Label Coordinate at" << it_label << " : " << *label_x << " " << *label_y << endl;
            switch (m_imageConfig->GetLabelType())
            {
            case LabelType::Regression:
                
                test_x = (test_x * (float)mat.cols - cropRect.x) / factor_rel_transform.at(0);
                test_y = (test_y * (float)mat.rows - cropRect.y) / factor_rel_transform.at(1);
                cout << "Cropped  TEST Coordinate  : " << test_x << " " << test_y << endl;

                *label_x = (*label_x - ((float)cropRect.x / mat.cols)) / ratio;
                *label_y = (*label_y - ((float)cropRect.y / mat.rows)) / ratio;
                cout << "Cropped  Coordinate  : " << *label_x << " " << *label_y << endl;

                // Set label to 0.0 or NaN, if cropped outside 
                if ((m_cropLandmark == CropModeLandmark::hard) || (m_cropLandmark == CropModeLandmark::both))
                {
                    if ((*label_x < m_LandmarkValueMin) || (*label_x > m_LandmarkValueMax) ||
                        (*label_y < m_LandmarkValueMin) || (*label_y > m_LandmarkValueMax))
                    {
                        *label_x = 0.0;
                        *label_y = 0.0;
                    }
                }

                break;
            case LabelType::Classification:
                break;
            default:
                ;
            }
        }

        // Crop Visibilities
        for (int it_label = 0; it_label < m_labelDimension; it_label ++)
        {
            if (m_cropVisibility == CropModeVisibility::none)
            {
                break;
            }

            // Check if current label is a Landmark
            if ((it_label + 1 < m_VisibilityLabels[0]) || (it_label + 1 >> m_VisibilityLabels[1]))
            {
                continue;
            }
        }
    }
    else if (m_imageConfig->GetElementType() == ElementType::tdouble)
    {
        // Change it to a template function
    }




    if ((m_hFlip && std::bernoulli_distribution()(*rng)) ||
        viewIndex >= 5)
    {
        cv::flip(mat, mat, 1);
    }

    m_rngs.push(std::move(rng));
}

CropTransformer::RatioJitterType CropTransformer::ParseJitterType(const std::string &src)
{
    cout << "CropTransformer::ParseJitterType " << src << '\n';
    if (src.empty() || AreEqualIgnoreCase(src, "none"))
    {
        return RatioJitterType::None;
    }

    if (AreEqualIgnoreCase(src, "uniratio"))
    {
        return RatioJitterType::UniRatio;
    }

    if (AreEqualIgnoreCase(src, "unilength"))
    {
        return RatioJitterType::UniLength;
    }

    if (AreEqualIgnoreCase(src, "uniarea"))
    {
        return RatioJitterType::UniArea;
    }

    RuntimeError("Invalid jitter type: %s.", src.c_str());
}

cv::Rect CropTransformer::GetCropRect(CropType type, int viewIndex, int crow, int ccol,
                                      double cropRatio, std::mt19937 &rng)
{
    // NOTE: crow , ccol = 224 x 224
    // cout << "MAT: " << crow << " " << ccol << '\n';
    assert(crow > 0);
    assert(ccol > 0);
    assert(0 < cropRatio && cropRatio <= 1.0);

    // Get square crop size that preserves aspect ratio.
    int cropSize = (int)(std::min(crow, ccol) * cropRatio);
    int cropSizeX = cropSize;
    int cropSizeY = cropSize;
    // Change aspect ratio, if this option is enabled.
    if (m_curAspectRatioRadius > 0)
    {
        double factor = 1.0 + UniRealT(-m_curAspectRatioRadius, m_curAspectRatioRadius)(rng);
        double area = cropSize * cropSize;
        double newArea = area * factor;
        cout << "Factor: "<< factor << endl;
        if (std::bernoulli_distribution()(rng))
        {
            cropSizeX = (int)std::sqrt(newArea);
            cropSizeY = (int)(area / cropSizeX);
        }
        else
        {
            cropSizeY = (int)std::sqrt(newArea);
            cropSizeX = (int)(area / cropSizeY);
        }
        // This clamping should be ok if jittering ratio is not too big.
        cropSizeX = std::min(cropSizeX, ccol);
        cropSizeY = std::min(cropSizeY, crow);
    }

    int xOff = -1;
    int yOff = -1;
    switch (type)
    {
    case CropType::Center:
        assert(viewIndex == 0);
        xOff = (ccol - cropSizeX) / 2;
        yOff = (crow - cropSizeY) / 2;
        break;
    case CropType::Random:
        assert(viewIndex == 0);
        xOff = UniIntT(0, ccol - cropSizeX)(rng);
        yOff = UniIntT(0, crow - cropSizeY)(rng);
        break;
    case CropType::MultiView10:
    {
        assert(0 <= viewIndex && viewIndex < 10);
        // 0 - 4: 4 corners + center crop. 5 - 9: same, but with a flip.
        int isubView = viewIndex % 5;
        switch (isubView)
        {
        // top-left
        case 0:
            xOff = 0;
            yOff = 0;
            break;
        // top-right
        case 1:
            xOff = ccol - cropSizeX;
            yOff = 0;
            break;
        // bottom-left
        case 2:
            xOff = 0;
            yOff = crow - cropSizeY;
            break;
        // bottom-right
        case 3:
            xOff = ccol - cropSizeX;
            yOff = crow - cropSizeY;
            break;
        // center
        case 4:
            xOff = (ccol - cropSizeX) / 2;
            yOff = (crow - cropSizeY) / 2;
            break;
        }
        break;
    }
    default:
        assert(false);
    }
    cout << "Rect " << xOff << " " << yOff << " " << cropSizeX << " " << cropSizeY << " CropRatio " << cropRatio << endl;

    assert(0 <= xOff && xOff <= ccol - cropSizeX);
    assert(0 <= yOff && yOff <= crow - cropSizeY);
    return cv::Rect(xOff, yOff, cropSizeX, cropSizeY);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ScaleTransformer::Initialize(TransformerPtr next,
                                  const ConfigParameters &readerConfig)
{
    ImageTransformerBase::Initialize(next, readerConfig);
    m_interpMap.emplace("nearest", cv::INTER_NEAREST);
    m_interpMap.emplace("linear", cv::INTER_LINEAR);
    m_interpMap.emplace("cubic", cv::INTER_CUBIC);
    m_interpMap.emplace("lanczos", cv::INTER_LANCZOS4);

    auto featureStreamIds = GetAppliedStreamIds();
    const auto &feature = GetInputStreams()[featureStreamIds[0]];
    m_dataType = feature->m_elementType == ElementType::tfloat ? CV_32F : CV_64F;

    InitFromConfig(readerConfig(feature->m_name));
}

void ScaleTransformer::InitFromConfig(const ConfigParameters &config)
{
    m_imgWidth = config(L"width");
    m_imgHeight = config(L"height");
    m_imgChannels = config(L"channels");

    size_t cfeat = m_imgWidth * m_imgHeight * m_imgChannels;
    if (cfeat == 0 || cfeat > std::numeric_limits<size_t>().max() / 2)
        RuntimeError("Invalid image dimensions.");

    m_interp.clear();
    std::stringstream ss{config(L"interpolations", "")};
    for (std::string token = ""; std::getline(ss, token, ':');)
    {
        // Explicit cast required for GCC.
        std::transform(token.begin(), token.end(), token.begin(),
                       (int (*) (int)) std::tolower);
        StrToIntMapT::const_iterator res = m_interpMap.find(token);
        if (res != m_interpMap.end())
            m_interp.push_back((*res).second);
    }

    if (m_interp.size() == 0)
        m_interp.push_back(cv::INTER_LINEAR);
}

void ScaleTransformer::Apply(size_t id, cv::Mat &mat, SequenceDataPtr labelPtr)
{
    cout << "ScaleTransformer::Apply, Image ID " << id << endl;
    UNUSED(id);
    // If matrix has not been converted to the right type, do it now as rescaling
    // requires floating point type.
    //
    if (mat.type() != CV_MAKETYPE(m_dataType, m_imgChannels))
    {
        mat.convertTo(mat, m_dataType);
    }

    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    auto index = UniIntT(0, static_cast<int>(m_interp.size()) - 1)(*rng);
    assert(m_interp.size() > 0);

    cv::resize(mat, mat, cv::Size((int)m_imgWidth, (int)m_imgHeight), 0, 0, m_interp[index]);

    m_rngs.push(std::move(rng));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MeanTransformer::Initialize(TransformerPtr next,
                                 const ConfigParameters &readerConfig)
{
    ImageTransformerBase::Initialize(next, readerConfig);

    auto featureStreamIds = GetAppliedStreamIds();
    InitFromConfig(readerConfig(GetInputStreams()[featureStreamIds[0]]->m_name));
}

void MeanTransformer::InitFromConfig(const ConfigParameters &config)
{
    std::wstring meanFile = config(L"meanFile", L"");
    if (meanFile.empty())
        m_meanImg.release();
    else
    {
        cv::FileStorage fs;
        // REVIEW alexeyk: this sort of defeats the purpose of using wstring at
        // all...  [fseide] no, only OpenCV has this problem.
        fs.open(msra::strfun::utf8(meanFile).c_str(), cv::FileStorage::READ);
        if (!fs.isOpened())
            RuntimeError("Could not open file: %ls", meanFile.c_str());
        fs["MeanImg"] >> m_meanImg;
        int cchan;
        fs["Channel"] >> cchan;
        int crow;
        fs["Row"] >> crow;
        int ccol;
        fs["Col"] >> ccol;
        if (cchan * crow * ccol !=
            m_meanImg.channels() * m_meanImg.rows * m_meanImg.cols)
            RuntimeError("Invalid data in file: %ls", meanFile.c_str());
        fs.release();
        m_meanImg = m_meanImg.reshape(cchan, crow);
    }
}

void MeanTransformer::Apply(size_t id, cv::Mat &mat, SequenceDataPtr labelPtr)
{
    cout << "MeanTransformer::Apply, Image ID " << id << endl;
    UNUSED(id);
    assert(m_meanImg.size() == cv::Size(0, 0) ||
           (m_meanImg.size() == mat.size() &&
            m_meanImg.channels() == mat.channels()));

    // REVIEW alexeyk: check type conversion (float/double).
    if (m_meanImg.size() == mat.size())
    {
        mat = mat - m_meanImg;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TransposeTransformer::Initialize(TransformerPtr next,
                                      const ConfigParameters &readerConfig)
{
    TransformerBase::Initialize(next, readerConfig);

    // Currently we only support a single stream.
    ImageConfigHelper config(readerConfig);
    size_t featureStreamId = config.GetFeatureStreamId();
    m_appliedStreamIds.push_back(featureStreamId);

    const auto &inputStreams = GetInputStreams();
    m_outputStreams.resize(inputStreams.size());
    std::copy(inputStreams.begin(), inputStreams.end(), m_outputStreams.begin());

    for (auto id : m_appliedStreamIds)
    {
        auto &stream = inputStreams[id];

        ImageDimensions dimensions(*stream->m_sampleLayout, HWC);

        // Changing from NHWC to NCHW (note: row-major notation)
        auto changedStream = std::make_shared<StreamDescription>(*stream);
        changedStream->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(CHW));
        m_outputStreams[id] = changedStream;
    }
}

SequenceDataPtr
TransposeTransformer::Apply(SequenceDataPtr inputSequence,
                            SequenceDataPtr &inputSequenceLabel,
                            const StreamDescription &inputStream,
                            const StreamDescription &outputStream)
{
    cout << "TransposeTransformer::Apply, Image ID " << inputSequence->m_id << endl;
    if (inputStream.m_elementType == ElementType::tdouble)
    {
        return TypedApply<double>(inputSequence, inputSequenceLabel, inputStream, outputStream);
    }

    if (inputStream.m_elementType == ElementType::tfloat)
    {
        return TypedApply<float>(inputSequence, inputSequenceLabel, inputStream, outputStream);
    }

    RuntimeError("Unsupported type");
}

// The class represents a sequence that owns an internal data buffer.
// Passed from the TransposeTransformer.
// TODO: Transposition potentially could be done in place (alexeyk: performance might be much worse than of out-of-place transpose).
struct DenseSequenceWithBuffer : DenseSequenceData
{
    std::vector<char> m_buffer;
};

template <class TElemType>
SequenceDataPtr TransposeTransformer::TypedApply(SequenceDataPtr sequence,
                                                 SequenceDataPtr &inputSequenceLabel,
                                                 const StreamDescription &inputStream,
                                                 const StreamDescription &outputStream)
{
    assert(inputStream.m_storageType == StorageType::dense);
    auto inputSequence = static_cast<DenseSequenceData&>(*sequence.get());
    assert(inputSequence.m_numberOfSamples == 1);
    assert(inputStream.m_sampleLayout->GetNumElements() == outputStream.m_sampleLayout->GetNumElements());

    size_t count = inputStream.m_sampleLayout->GetNumElements() * GetSizeByType(inputStream.m_elementType);

    auto result = std::make_shared<DenseSequenceWithBuffer>();
    result->m_buffer.resize(count);

    ImageDimensions dimensions(*inputStream.m_sampleLayout, ImageLayoutKind::HWC);
    size_t rowCount = dimensions.m_height * dimensions.m_width;
    size_t channelCount = dimensions.m_numChannels;

    auto src = reinterpret_cast<TElemType*>(inputSequence.m_data);
    auto dst = reinterpret_cast<TElemType*>(result->m_buffer.data());

    for (size_t irow = 0; irow < rowCount; irow++)
    {
        for (size_t icol = 0; icol < channelCount; icol++)
        {
            dst[icol * rowCount + irow] = src[irow * channelCount + icol];
        }
    }

    result->m_id = sequence->m_id;
    result->m_sampleLayout = outputStream.m_sampleLayout;
    result->m_data = result->m_buffer.data();
    result->m_numberOfSamples = inputSequence.m_numberOfSamples;
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void IntensityTransformer::Initialize(TransformerPtr next,
                                 const ConfigParameters &readerConfig)
{
    ImageTransformerBase::Initialize(next, readerConfig);

    auto featureStreamIds = GetAppliedStreamIds();
    InitFromConfig(readerConfig(GetInputStreams()[featureStreamIds[0]]->m_name));
}

void IntensityTransformer::InitFromConfig(const ConfigParameters &config)
{
    m_stdDev = config(L"intensityStdDev", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    std::wstring intFile = config(L"intensityFile", L"");
    if (intFile.empty())
    {
        m_eigVal.release();
        m_eigVec.release();
    }
    else
    {
        cv::FileStorage fs;
        fs.open(msra::strfun::utf8(intFile).c_str(), cv::FileStorage::READ);
        if (!fs.isOpened())
            RuntimeError("Could not open file: %ls", intFile.c_str());
        fs["EigVal"] >> m_eigVal;
        if (m_eigVal.rows != 1 || m_eigVal.cols != 3 || m_eigVal.channels() != 1)
            RuntimeError("Invalid EigVal data in file: %ls", intFile.c_str());
        fs["EigVec"] >> m_eigVec;
        if (m_eigVec.rows != 3 || m_eigVec.cols != 3 || m_eigVec.channels() != 1)
            RuntimeError("Invalid EigVec data in file: %ls", intFile.c_str());
        fs.release();
    }
}

void IntensityTransformer::StartEpoch(const EpochConfiguration &config)
{
    m_curStdDev = m_stdDev[config.m_epochIndex];

    ImageTransformerBase::StartEpoch(config);
}

void IntensityTransformer::Apply(size_t id, cv::Mat &mat, SequenceDataPtr labelPtr)
{
    cout << "IntensityTransformer::Apply, Image ID " << id << endl;
    UNUSED(id);

    if (m_eigVal.empty() || m_eigVec.empty() || m_curStdDev == 0)
        return;

    if (mat.type() == CV_64FC(mat.channels()))
        Apply<double>(mat);
    else if (mat.type() == CV_32FC(mat.channels()))
        Apply<float>(mat);
    else
        RuntimeError("Unsupported type");
}

template <typename ElemType>
void IntensityTransformer::Apply(cv::Mat &mat)
{
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); } );

    // Using single precision as EigVal and EigVec matrices are single precision.
    std::normal_distribution<float> d(0, (float)m_curStdDev);
    cv::Mat alphas(1, 3, CV_32FC1);
    assert(m_eigVal.rows == 1 && m_eigVec.cols == 3);
    alphas.at<float>(0) = d(*rng) * m_eigVal.at<float>(0);
    alphas.at<float>(1) = d(*rng) * m_eigVal.at<float>(1);
    alphas.at<float>(2) = d(*rng) * m_eigVal.at<float>(2);
    m_rngs.push(std::move(rng));

    assert(m_eigVec.rows == 3 && m_eigVec.cols == 3);

    cv::Mat shifts = m_eigVec * alphas.t();

    // For multi-channel images data is in BGR format.
    size_t cdst = mat.rows * mat.cols * mat.channels();
    ElemType* pdstBase = reinterpret_cast<ElemType*>(mat.data);
    for (ElemType* pdst = pdstBase; pdst < pdstBase + cdst;)
    {
        for (int c = 0; c < mat.channels(); c++)
        {
            float shift = shifts.at<float>(mat.channels() - c - 1);
            *pdst = std::min(std::max(*pdst + shift, (ElemType)0), (ElemType)255);
            pdst++;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ColorTransformer::Initialize(TransformerPtr next, const ConfigParameters &readerConfig)
{
    ImageTransformerBase::Initialize(next, readerConfig);

    auto featureStreamIds = GetAppliedStreamIds();
    InitFromConfig(readerConfig(GetInputStreams()[featureStreamIds[0]]->m_name));
}

void ColorTransformer::InitFromConfig(const ConfigParameters &config)
{
    m_brightnessRadius = config(L"brightnessRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    m_contrastRadius = config(L"contrastRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
    m_saturationRadius = config(L"saturationRadius", ConfigParameters::Array(doubleargvector(vector<double>{0.0})));
}

void ColorTransformer::StartEpoch(const EpochConfiguration &config)
{
    m_curBrightnessRadius = m_brightnessRadius[config.m_epochIndex];
    if (!(0 <= m_curBrightnessRadius && m_curBrightnessRadius <= 1.0))
        InvalidArgument("brightnessRadius must be >= 0.0 and <= 1.0");

    m_curContrastRadius = m_contrastRadius[config.m_epochIndex];
    if (!(0 <= m_curContrastRadius && m_curContrastRadius <= 1.0))
        InvalidArgument("contrastRadius must be >= 0.0 and <= 1.0");

    m_curSaturationRadius = m_saturationRadius[config.m_epochIndex];
    if (!(0 <= m_curSaturationRadius && m_curSaturationRadius <= 1.0))
        InvalidArgument("saturationRadius must be >= 0.0 and <= 1.0");

    ImageTransformerBase::StartEpoch(config);
}

void ColorTransformer::Apply(size_t id, cv::Mat &mat, SequenceDataPtr labelPtr)
{
    cout << "ColorTransformer::Apply, Image ID " << id << endl;
    UNUSED(id);

    if (m_curBrightnessRadius == 0 && m_curContrastRadius == 0 && m_curSaturationRadius == 0)
        return;

    if (mat.type() == CV_64FC(mat.channels()))
        Apply<double>(mat);
    else if (mat.type() == CV_32FC(mat.channels()))
        Apply<float>(mat);
    else
        RuntimeError("Unsupported type");
}

template <typename ElemType>
void ColorTransformer::Apply(cv::Mat &mat)
{
    auto seed = GetSeed();
    auto rng = m_rngs.pop_or_create([seed]() { return std::make_unique<std::mt19937>(seed); });

    if (m_curBrightnessRadius > 0 || m_curContrastRadius > 0)
    {
        // To change brightness and/or contrast the following standard transformation is used:
        // Xij = alpha * Xij + beta, where
        // alpha is a contrast adjustment and beta - brightness adjustment.
        ElemType beta = 0;
        if (m_curBrightnessRadius > 0)
        {
            UniRealT d(-m_curBrightnessRadius, m_curBrightnessRadius);
            // Compute mean value of the image.
            cv::Scalar imgMean = cv::sum(cv::sum(mat));
            // Compute beta as a fraction of the mean.
            beta = (ElemType)(d(*rng) * imgMean[0] / (mat.rows * mat.cols * mat.channels()));
        }

        ElemType alpha = 1;
        if (m_curContrastRadius > 0)
        {
            UniRealT d(-m_curContrastRadius, m_curContrastRadius);
            alpha = (ElemType)(1 + d(*rng));
        }

        // Could potentially use mat.convertTo(mat, -1, alpha, beta) 
        // but it does not do range checking for single/double precision matrix. saturate_cast won't work either.
        size_t count = mat.rows * mat.cols * mat.channels();
        ElemType* pbase = reinterpret_cast<ElemType*>(mat.data);
        for (ElemType* p = pbase; p < pbase + count; p++)
        {
            *p = std::min(std::max(*p * alpha + beta, (ElemType)0), (ElemType)255);
        }
    }

    if (m_curSaturationRadius > 0 && mat.channels() == 3)
    {
        UniRealT d(-m_curSaturationRadius, m_curSaturationRadius);
        double ratio = 1.0 + d(*rng);
        assert(0 <= ratio && ratio <= 2);

        auto hsv = m_hsvTemp.pop_or_create([]() { return std::make_unique<cv::Mat>(); });

        // To change saturation, we need to convert the image to HSV format first,
        // the change S channgel and convert the image back to BGR format.
        cv::cvtColor(mat, *hsv, CV_BGR2HSV);
        assert(hsv->rows == mat.rows && hsv->cols == mat.cols);
        size_t count = hsv->rows * hsv->cols * mat.channels();
        ElemType* phsvBase = reinterpret_cast<ElemType*>(hsv->data);
        for (ElemType* phsv = phsvBase; phsv < phsvBase + count; phsv += 3)
        {
            const int HsvIndex = 1;
            phsv[HsvIndex] = std::min((ElemType)(phsv[HsvIndex] * ratio), (ElemType)1);
        }
        cv::cvtColor(*hsv, mat, CV_HSV2BGR);

        m_hsvTemp.push(std::move(hsv));
    }

    m_rngs.push(std::move(rng));
}

}}}
