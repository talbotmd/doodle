Sketchy Database - General Information

Photo Identification
  Photos are identified by their ImageNet ID. This generally looks like:
  n1111111_2222, where n1111111 is the synset or category ID and
  2222 is the image ID within that synset.

Sketch Identification
  Sketch IDs start with the photo ID, but each has an additional number
  added to the end. For example, the five sketches of photo n1111111_2222
  will be labeled n1111111_2222-1, n1111111_2222-2, ..., n1111111_2222-5.

Sketch Validity Labels
  Sketches were manually labeled for validity. However, invalid sketches
  were not removed from the dataset (since each may still be useful for
  some task).

  Error - Something is very wrong with the sketch. It may be completely
    incorrect and/or a squiggle. There is little use for these outside
    perhaps crowd behavior research.
  Ambiguous - The reviewer deemed the sketch too poor quality to identify
    as the subject object. For example, a pizza drawn as a circle with no
    'slices', 'crust', or 'toppings'. However, these sketches may still
    approximate the correct shape and/or pose.
  Pose - The reviewer deemed the sketch identifiable, but not in a correct
    pose or perspective. This is of particular importance to fine-grained
    sketch based image retrieval.
  Context - The artist included environmental details that were not part of
    the subject object, such as 'water ripples' around a duck or a flower
    on which a bee rests.

Info Contents:
  stats.csv - a table containing detailed information about
    each sketch within the dataset
  strokes.csv - a table containing stroke and time related
    statistics for all sketches within the dataset (used to
    generated figures in the paper)
  testset.txt - a listing of 10 photographs withheld from
    training; organized alphabetically by category label
  urls.txt - a listing of all photo IDs and their original URLs
  invalid-error.txt - a listing of all sketches that were deemed invalid 
    due to error
  invalid-ambiguous.txt - a listing of all sketches that were deemed 
    invalid due to ambiguity
  invalid-pose.txt - a listing of all sketches that were deemed invalid 
    due to pose
  invalid-context.txt - a lsiting of all sketches that were deemed invalid 
    due to context
