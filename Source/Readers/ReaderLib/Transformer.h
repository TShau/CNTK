//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

// Defines a data transformation interface.
// Transformers are responsible for doing custom transformation of sequences.
// For example for images, there could be scale, crop, or median transformation.
class Transformer
{
public:
    // Starts a new epoch. Some transformers have to change their configuration
    // based on the epoch.
    virtual void StartEpoch(const EpochConfiguration &config) = 0;

    // Transformers are applied on a particular input stream - this method should describe
    // how inputStream is transformed to the output stream (return value)
    virtual StreamDescription Transform(const StreamDescription& inputStream) = 0;

    // This method should describe how input sequences is transformed to the output sequence.
    virtual SequenceDataPtr Transform(SequenceDataPtr inputSequence, SequenceDataPtr label_sequence) = 0;

    virtual ~Transformer()
    {
    }
};

}}}
