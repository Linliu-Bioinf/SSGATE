#!/bin/env python3
import torch









def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdims=True)[0]
#    tf.summary.scalar("hardest_positive_dist", torch.mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = torch.max(pairwise_dist, axis=1, keepdims=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdims=True)[0]
#    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    combs = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss = combs[torch.gt(combs, 0.0)]

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def pairwise_distances(z, squared=False):
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(z, z.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diagonal(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.tensor(0.0))

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    return distances


def get_anchor_positive_triplet_mask(labels):
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    indices_equal = torch.eye(labels.shape[0], dtype = torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)

    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask


def get_anchor_negative_triplet_mask(labels):
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    mask = torch.logical_not(labels_equal)
    return mask


def get_triplet_mask(labels):
    # Check that i, j and k are distinct
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    indices_equal = torch.eye(labels.shape[0], dtype = torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)

    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
    # Combine the two masks
    mask = torch.logical_and(distinct_indices, valid_labels)

    return mask



















