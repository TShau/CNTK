//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <set>

#include "Transformer.h"
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

class TransformerBase : public Transformer
{
public:
    // Initializes the transformer.
    virtual void Initialize(TransformerPtr next,
                            const ConfigParameters &) override
    {
        cout << "Base Initialize" << endl;
        m_next = next;
        m_inputStreams = m_next->GetStreamDescriptions();
    }

    // Sets configuration for the current epoch.
    virtual void StartEpoch(const EpochConfiguration &config) override
    {
        assert(m_next != nullptr);
        m_next->StartEpoch(config);
    }

    // Description of streams that the transformer provides.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return this->GetOutputStreams();
    }

    // Gets next sequences up to a maximum count of samples.
    // Sequences contains data for all streams.
    virtual Sequences GetNextSequences(size_t sampleCount) override
    {
        assert(m_next != nullptr);
        Sequences samples = m_next->GetNextSequences(sampleCount);

        if (samples.m_data.empty())
        {
            return samples;
        }

        const auto &appliedStreamIds = GetAppliedStreamIds();
        const auto &outputStreams = GetOutputStreams();

        // TODO: Move parallelization on the outer loop with collapse.

        // NOTE: allpiedStreamIds only 0 -> change it if Transformation with Regression?
        for (int j = 0; j < appliedStreamIds.size(); ++j)
        {
            size_t streamId = appliedStreamIds[j];
            //auto& allSamples = samples.m_data[streamId];

            
            auto& allSamples = samples.m_data[0];
            auto& allLabels  = samples.m_data[1];
//#pragma omp parallel for schedule(static)
            for (int i = 0; i < allSamples.size(); ++i)
            {
                allSamples[i] = Apply(allSamples[i], allLabels[i], *m_inputStreams[streamId], *outputStreams[streamId]);
            }
        }
        return samples;
    }

protected:
    virtual const std::vector<StreamId> &GetAppliedStreamIds() const = 0;
    virtual const std::vector<StreamDescriptionPtr> &GetOutputStreams() const
    {
        return m_inputStreams;
    }

    const std::vector<StreamDescriptionPtr> &GetInputStreams()
    {
        return m_inputStreams;
    }

private:
    // Applies transformation to the sequence.
    virtual SequenceDataPtr Apply(SequenceDataPtr inputSequence,
                                  SequenceDataPtr &inputSequenceLabel,
                                  const StreamDescription &inputStream,
                                  const StreamDescription &outputStream) = 0;

    TransformerPtr m_next;
    std::vector<StreamId> m_featureStreamIds;
    std::vector<StreamDescriptionPtr> m_inputStreams;
};

}}}
