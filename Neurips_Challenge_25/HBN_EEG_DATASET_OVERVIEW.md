# HBN-EEG Dataset Overview

## Dataset Summary

The **Healthy Brain Network (HBN) EEG Dataset** is a large-scale, multi-release collection of high-density EEG recordings designed to advance understanding of child and adolescent mental health through neuroimaging, behavioral, and genetic data analysis.

### Key Statistics
- **Total Participants**: >3,000 (ages 5-21 years)
- **EEG Channels**: 129 high-density electrodes
- **Data Releases**: 12 total (11 public + 1 competition-reserved)
- **Total Dataset Size**: Multi-terabyte scale
- **Format**: BIDS (Brain Imaging Data Structure) standard
- **License**: CC-BY-SA 4.0 (research releases), CC-BY-NC-SA 4.0 (NC release)

## Dataset Releases Breakdown

### Public Releases (Training/Validation Data)
| Release | Dataset ID | Subjects | S3 URI | Usage |
|---------|------------|----------|---------|-------|
| Release 1 | DS005505 | 136 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1` | Training |
| Release 2 | DS005506 | 152 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R2` | Training |
| Release 3 | DS005507 | 183 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R3` | Training |
| Release 4 | DS005508 | 324 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R4` | Training |
| **Release 5** | **DS005509** | **330** | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R5` | **Validation** |
| Release 6 | DS005510 | 134 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R6` | Training |
| Release 7 | DS005511 | 381 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R7` | Training |
| Release 8 | DS005512 | 257 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R8` | Training |
| Release 9 | DS005514 | 295 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R9` | Training |
| Release 10 | DS005515 | 533 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R10` | Training |
| Release 11 | DS005516 | 430 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R11` | Training |

**Training Total**: ~2,635 subjects (Releases 1-4, 6-11)

### Reserved/Special Releases
| Release | Dataset ID | Subjects | Access | Purpose |
|---------|------------|----------|---------|---------|
| **Release 12** | *Unreleased* | *Unknown* | Competition only | **Test set** |
| **Release NC** | *Not on OpenNeuro* | 458 | Research only | Non-commercial use |

## Task Structure

The HBN-EEG dataset includes **6 distinct cognitive tasks** categorized into passive and active paradigms:

### Passive Tasks (No User Input Required)
1. **Resting State (RS)**
   - **Description**: Participants rest with eyes open/closed, fixating on central cross
   - **Purpose**: Baseline brain activity measurement
   - **Event Markers**: Eye state changes, fixation periods

2. **Surround Suppression (SuS)**
   - **Description**: Viewing flashing peripheral disks with contrasting backgrounds
   - **Purpose**: Visual processing and suppression mechanisms
   - **Event Markers**: Stimulus presentations, condition changes

3. **Movie Watching (MW)**
   - **Description**: Watching four short movies with different themes
   - **Movies**: DespicableMe, DiaryOfAWimpyKid, FunwithFractals, ThePresent
   - **Purpose**: Natural stimulus processing, attention, emotion
   - **Event Markers**: Movie start/stop times, scene changes

### Active Tasks (User Input Required)
4. **Contrast Change Detection (CCD)**
   - **Description**: Identifying flickering disks with dominant contrast changes
   - **Purpose**: Visual attention, decision-making, response time measurement
   - **Behavioral Data**: Response times, accuracy, feedback

5. **Sequence Learning (SL)**
   - **Description**: Memorizing and reproducing sequences of flashed circles
   - **Variants**: 6-target and 8-target sequences (age-appropriate)
   - **Purpose**: Working memory, sequence learning, motor planning
   - **Behavioral Data**: Sequence accuracy, learning curves

6. **Symbol Search (SyS)**
   - **Description**: Computerized version of WISC-IV subtest
   - **Purpose**: Visual processing speed, attention, symbol recognition
   - **Behavioral Data**: Search accuracy, response times

## Data Content and Structure

### EEG Recordings
- **Electrode System**: 129-channel high-density EEG
- **Data Format**: EEGLAB .set files
- **Sampling Rate**: Variable (task-dependent)
- **Reference**: Standard EEG montage
- **Quality Control**: Minimal preprocessing, corruption checks

### Behavioral Data
- **Response Times**: Millisecond precision for active tasks
- **Accuracy Measures**: Hit rates, error types
- **Task Performance**: Success rates, learning metrics
- **Integration**: Included in events.tsv files with EEG data

### Psychopathology Factors
Derived from **Child Behavior Checklist (CBCL)** questionnaire:

1. **P-Factor**: General psychopathology dimension
2. **Internalizing**: Anxiety, depression, withdrawal symptoms
3. **Externalizing**: ADHD, conduct problems, aggression
4. **Attention**: Attention deficit and hyperactivity measures

### Demographic Information
- **Age Range**: 5-21 years (children and adolescents)
- **Sample Size**: >3,000 participants
- **Additional Factors**: Gender, handedness (Edinburgh Inventory)

## Technical Specifications

### Data Standards
- **BIDS Version**: 1.9.0
- **HED Version**: 8.3.0 (Hierarchical Event Descriptors)
- **File Format**: .set (EEGLAB), .tsv (events/participants)
- **Metadata**: Comprehensive JSON sidecar files

### Event Annotation
- **HED Tags**: Systematic event description for meta-analysis
- **Event Types**: Stimulus presentations, responses, state changes
- **Temporal Precision**: Millisecond-level event timing
- **Behavioral Integration**: Response events synchronized with EEG

### Quality Assurance
- **Data Integrity**: Corruption checks, file validation
- **Event Completeness**: All tasks have required event markers
- **Preprocessing Ready**: Minimal processing, ready for analysis
- **Quality Metrics**: Available in participants.tsv

## Competition Usage

### Data Splits for NeurIPS 2025 Challenge
- **Training**: Releases 1-4, 6-11 (~2,635 subjects)
- **Validation**: Release 5 (330 subjects)
- **Test**: Release 12 (unreleased, competition evaluation)

### Challenge Tasks
1. **Challenge 1**: Cross-task transfer learning
   - Train on passive tasks → Predict active task performance
   - Zero-shot generalization to new subjects and tasks

2. **Challenge 2**: Psychopathology factor prediction
   - Predict mental health dimensions from EEG
   - Multi-factor regression (P-factor, internalizing, externalizing, attention)

## Access and Usage

### Data Access
- **Primary Portal**: NEMAR (NeuroElectroMagnetic Analysis & Repositories)
- **Secondary**: OpenNeuro integration
- **Download Options**: Direct download, AWS S3 access
- **Processing**: Neuroscience Gateway (NSG) integration

### Licensing and Ethics
- **License**: CC-BY-SA 4.0 (most releases)
- **Ethics**: Chesapeake Institutional Review Board approval
- **Consent**: Written informed consent from participants/guardians
- **Anonymization**: NIMH Global Unique Identifier Tool (GUID)

### Citation Requirements
**Primary Citation**:
```
Shirazi, S.Y., Franco, A., Hoffmann, M.S., et al. (2024). 
HBN-EEG: The FAIR implementation of the healthy brain network (HBN) 
electroencephalography dataset. bioRxiv. 
https://doi.org/10.1101/2024.10.03.615261
```

**Original HBN Citation**:
```
Alexander, L.M., et al. (2017). 
An open resource for transdiagnostic research in pediatric mental health 
and learning disorders. Sci Data 4, 170181. 
https://dx.doi.org/10.1038/sdata.2017.181
```

## Future Enhancements

### Planned Additions
1. **Personalized EEG Electrode Locations**
   - Individual electrode positioning data
   - Improved spatial analysis capabilities

2. **Personalized Lead Field Matrix**
   - Subject-specific forward models
   - Enhanced source localization

3. **Eye-Tracking Data**
   - Synchronized gaze tracking during tasks
   - Visual attention analysis integration

### Research Applications
- **Cross-task generalization studies**
- **Mental health biomarker discovery**
- **Developmental neuroscience research**
- **Brain-computer interface development**
- **Foundation model training and evaluation**

## Technical Support

### Resources
- **Documentation**: Comprehensive README and metadata files
- **Community**: Discussion forums and issue tracking
- **Computational**: NSG processing pipeline support
- **Updates**: Regular dataset version releases

### Funding Acknowledgment
- **NIH/NIMH**: R01MH125934 (BIDS preparation)
- **NIMH**: R24MH120037 (NEMAR platform)
- **Child Mind Institute**: Primary data collection
- **Multiple Donors**: See HBN funding page

---

*This dataset represents one of the largest publicly available pediatric EEG collections, specifically designed to advance understanding of mental health through large-scale neuroimaging analysis and machine learning applications.*
