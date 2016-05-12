//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <limits>
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ImageDataDeserializer::ClassificationLabelGenerator
{
public:
    virtual void CreateLabelFor(const ImageSequenceDescription& desc, SparseSequenceData& data) = 0;
    virtual ~ClassificationLabelGenerator() { }
};

// A helper class to generate a typed label in a sparse format.
// A label is just a category/class the image belongs to.
// It is represented as a array indexed by the category with zero values for all categories the image does not belong to, 
// and a single one for a category it belongs to: [ 0, .. 0.. 1 .. 0 ]
// The class is parameterized because the representation of 1 is type specific.
template <class TElement>
class TypedClassificationLabelGenerator : public ImageDataDeserializer::ClassificationLabelGenerator
{
public:
    TypedClassificationLabelGenerator(size_t labelDimension) : m_value(1), m_indices(labelDimension)
    {
        if (labelDimension > numeric_limits<IndexType>::max())
        {
            RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
                "value (%" PRIu64 ")\n", labelDimension, (size_t)numeric_limits<IndexType>::max());
        }
        iota(m_indices.begin(), m_indices.end(), 0);
    }

    virtual void CreateLabelFor(const ImageDataDeserializer::ImageSequenceDescription& desc, SparseSequenceData& data) override
    {
        data.m_nnzCounts.resize(1);
        data.m_nnzCounts[0] = 1;
        data.m_totalNnzCount = 1;
        data.m_data = &m_value;
        data.m_indices = &(m_indices[desc.m_classId]);
    }

private:
    TElement m_value;
    vector<IndexType> m_indices;
};

// Used to keep track of the image. Accessed only using DenseSequenceData interface.
struct DeserializedImage : DenseSequenceData
{
    cv::Mat m_image;
};

// For image, chunks correspond to a single image.
class ImageDataDeserializer::ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk>
{
    ImageSequenceDescription m_description;
    ImageDataDeserializer& m_parent;

public:
    ImageChunk(ImageSequenceDescription& description, ImageDataDeserializer& parent)
        : m_description(description), m_parent(parent)
    {
    }

    virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
    {
        assert(sequenceId == m_description.m_id);
        const auto& imageSequence = m_description;

        auto image = std::make_shared<DeserializedImage>();
        image->m_image = std::move(m_parent.ReadImage(m_description.m_id, imageSequence.m_path, m_parent.m_grayscale));
        auto& cvImage = image->m_image;

        if (!cvImage.data)
        {
            RuntimeError("Cannot open file '%s'", imageSequence.m_path.c_str());
        }

        // Convert element type.
        int dataType = m_parent.m_featureElementType == ElementType::tfloat ? CV_32F : CV_64F;
        if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
        {
            cvImage.convertTo(cvImage, dataType);
        }

        if (!cvImage.isContinuous())
        {
            cvImage = cvImage.clone();
        }
        assert(cvImage.isContinuous());

        image->m_data = image->m_image.data;
        ImageDimensions dimensions(cvImage.cols, cvImage.rows, cvImage.channels());
        image->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
        image->m_id = sequenceId;
        image->m_numberOfSamples = 1;
        image->m_chunk = shared_from_this();
        result.push_back(image);

        SequenceDataPtr label;
        LabelType type = m_parent.m_labelType;
        if (type == LabelType::Classification)
        {
            auto temp_label = std::make_shared<SparseSequenceData>(); // temp variable needed to avoid dynamic cast in the next line
            m_parent.m_classificationLabelGenerator->CreateLabelFor(imageSequence, *temp_label);
            label = temp_label;
        }
        else
        {
            assert(type == LabelType::Regression);
            label = std::make_shared<DenseSequenceData>();
            label->m_data = m_parent.m_featureElementType == ElementType::tfloat ? (void*)imageSequence.m_floatlabel.data() : (void*)imageSequence.m_doublelabel.data();
        }

        label->m_chunk = shared_from_this();
        label->m_numberOfSamples = 1;
        result.push_back(label);
    }
};

ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
{
    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);
    m_grayscale = configHelper.UseGrayscale();
    const auto& label = m_streams[configHelper.GetLabelStreamId()];
    const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

    // Expect data in HWC.
    ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
    feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));

    m_labelType = configHelper.GetLabelType();
    switch (m_labelType)
    {
    case LabelType::Classification: label->m_storageType = StorageType::sparse_csc; break;
    case LabelType::Regression: label->m_storageType = StorageType::dense; break;
    default: RuntimeError("Unsupported label type. Must be Classification (default) or Regression.");
    }

    feature->m_storageType = StorageType::dense;

    m_featureElementType = feature->m_elementType;
    size_t labelDimension = label->m_sampleLayout->GetDim(0);

    if (label->m_elementType == ElementType::tfloat)
    {
        /* DO EVEN MORE STUFF */
        m_classificationLabelGenerator = std::make_shared<TypedClassificationLabelGenerator<float>>(labelDimension);
    }
    else if (label->m_elementType == ElementType::tdouble)
    {
        /* DO EVEN MORE STUFF */
        m_classificationLabelGenerator = std::make_shared<TypedClassificationLabelGenerator<double>>(labelDimension);
    }
    else
    {
        RuntimeError("Unsupported label element type '%d'.", (int)label->m_elementType);
    }

    CreateSequenceDescriptions(configHelper.GetMapPath(), labelDimension, configHelper);
}

// Descriptions of chunks exposed by the image reader.
ChunkDescriptions ImageDataDeserializer::GetChunkDescriptions()
{
    ChunkDescriptions result;
    result.reserve(m_imageSequences.size());
    for (auto const& s : m_imageSequences)
    {
        auto chunk = std::make_shared<ChunkDescription>();
        chunk->m_id = s.m_chunkId;
        chunk->m_numberOfSamples = 1;
        chunk->m_numberOfSequences = 1;
        result.push_back(chunk);
    }

    return result;
}

void ImageDataDeserializer::GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result)
{
    // Currently a single sequence per chunk.
    result.push_back(m_imageSequences[chunkId]);
}

void ImageDataDeserializer::CreateSequenceDescriptions(std::string mapPath, size_t labelDimension, const ImageConfigHelper& config)
{
    std::ifstream mapFile(mapPath);
    if (!mapFile)
    {
        RuntimeError("Could not open %s for reading.", mapPath.c_str());
    }

    size_t itemsPerLine = config.IsMultiViewCrop() ? 10 : 1;
    size_t curId = 0;
    std::string line;
    PathReaderMap knownReaders;
    ImageSequenceDescription description;

    description.m_numberOfSamples = 1;
    description.m_isValid = true;

    for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
    {
        std::stringstream ss(line);
        std::string imagePath;
        std::string read;
        std::vector<float> float_vec;
        std::vector<double> double_vec;
        size_t cid = 0;
        float f = 0.0f;
        double d = 0.0;

        switch (m_labelType)
        {
        case LabelType::Classification:
            if (!std::getline(ss, imagePath, '\t') || !std::getline(ss, read, '\t'))
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());

            char* eptr;
            errno = 0;
            cid = strtoull(read.c_str(), &eptr, 10);
            if (read.c_str() == eptr || errno == ERANGE)
                RuntimeError("Cannot parse label value on line %" PRIu64 ", second column, in file %s.", lineIndex, mapPath.c_str());

            if (cid >= labelDimension)
            {
                RuntimeError(
                    "Image '%s' has invalid class id '%" PRIu64 "'. Expected label dimension is '%" PRIu64 "'. Line %" PRIu64 " in file %s.",
                    imagePath.c_str(), cid, labelDimension, lineIndex, mapPath.c_str());
            } break;

        case LabelType::Regression: 
            if (!std::getline(ss, imagePath, '\t'))
                RuntimeError("Could not read map file, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
            for (size_t i = 0; i < labelDimension; ++i)
            {
                auto invoke_error = [=](){RuntimeError("Cannot parse label value on line %" PRIu64 ", column %d, in file %s.", lineIndex, i, mapPath.c_str()); };
                if (!std::getline(ss, read, '\t'))
                    invoke_error();
                char* eptr;
                errno = 0;
                d = strtod(read.c_str(), &eptr);
                if (read.c_str() == eptr || errno == ERANGE)
                    invoke_error();
                f = static_cast<float>(d);        
                
                if (m_featureElementType == ElementType::tfloat)
                    float_vec.push_back(f);
                else
                    double_vec.push_back(d);
            }
            break;

        default: RuntimeError("Unsupported label type. Must be Classification (default) or Regression.");
        }
                
        for (size_t start = curId; curId < start + itemsPerLine; curId++)
        {
            description.m_id = curId;
            description.m_chunkId = curId;
            description.m_path = imagePath;
            
            // two out of the three lines below will effectively do nothing
            description.m_classId = cid;
            description.m_floatlabel = float_vec;
            description.m_doublelabel = double_vec; 
            
            description.m_key.m_sequence = description.m_id;
            description.m_key.m_sample = 0;

            m_imageSequences.push_back(description);
            RegisterByteReader(description.m_id, description.m_path, knownReaders);
        }
    }
}

ChunkPtr ImageDataDeserializer::GetChunk(size_t chunkId)
{
    auto sequenceDescription = m_imageSequences[chunkId];
    return std::make_shared<ImageChunk>(sequenceDescription, *this);
}

void ImageDataDeserializer::RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders)
{
    assert(!path.empty());

    auto atPos = path.find_first_of('@');
    // Is it container or plain image file?
    if (atPos == std::string::npos)
        return;
    // REVIEW alexeyk: only .zip container support for now.
#ifdef USE_ZIP
    assert(atPos > 0);
    assert(atPos + 1 < path.length());
    auto containerPath = path.substr(0, atPos);
    // skip @ symbol and path separator (/ or \)
    auto itemPath = path.substr(atPos + 2);
    // zlib only supports / as path separator.
    std::replace(begin(itemPath), end(itemPath), '\\', '/');
    std::shared_ptr<ByteReader> reader;
    auto r = knownReaders.find(containerPath);
    if (r == knownReaders.end())
    {
        reader = std::make_shared<ZipByteReader>(containerPath);
        knownReaders[containerPath] = reader;
    }
    else
    {
        reader = (*r).second;
    }
    reader->Register(seqId, itemPath);
    m_readers[seqId] = reader;
#else
    UNUSED(seqId);
    UNUSED(knownReaders);
    RuntimeError("The code is built without zip container support. Only plain image files are supported.");
#endif
}

cv::Mat ImageDataDeserializer::ReadImage(size_t seqId, const std::string& path, bool grayscale)
{
    assert(!path.empty());

    ImageDataDeserializer::SeqReaderMap::const_iterator r;
    if (m_readers.empty() || (r = m_readers.find(seqId)) == m_readers.end())
        return m_defaultReader.Read(seqId, path, grayscale);
    return (*r).second->Read(seqId, path, grayscale);
}

cv::Mat FileByteReader::Read(size_t, const std::string& path, bool grayscale)
{
    assert(!path.empty());

    return cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}
}}}
