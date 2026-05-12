/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_AVL_H
#define DDSRT_AVL_H

/* The tree library never performs memory allocations or deallocations internally.

   - Treedef_t: defines the properties of the tree, offsets,
     comparison functions, augmented structures, flags -- these are
     related to the code/data structure in which the tree is embedded,
     and in nearly all cases known at compile time.
   - avlTree_t: represents the tree, i.e., pointer to the root.
   - avlNode_t: contains the administrative data for a single node in
     the tree.

   For a tree node:
     struct T {
       avlNode_t avlnode;
       int key;
     };
   by definition, avlnodeoffset == offsetof(struct T, avlnode) and
   keyoffset = offsetof(struct T, key). The user of the library only
   ever deals in pointers to (in this case) struct T, never with
   pointers to the avlNode_t, and the compare function operations on
   pointers to keys, in this case pointers to "int"s. If you wish, you
   can also do: keyoffset = 0, in which case the compare function
   would be operating on struct T's.

   The compare function is assumed to behave just like all compare
   functions in the C library: < 0, =0, >0 for left argument less
   than, equal to or greater than the right argument.

   The "augment" function is automatically called whenever some of the
   children of a node change, as well as when the "augment" function
   has been called on some of the children. It allows you to maintain
   a "summary" of the subtree -- currently only used in ddsi2e, in one
   spot.

   Trees come in various "variants", configured through "treedef"
   flags:
   - direct/indirect key: direct meaning the key value is embedded in
     the structure containing the avlNode_t, indirect meaning a
     pointer to the key value is. The compare function doesn't deal
     with tree nodes, but with key values.
   - re-entrant: in the style of the C library, meaning, the
     comparison function gets a user-supplied 3rd argument (in
     particular used by mmstat).
   - unique keys/duplicate keys: when keys must be unique, some
     optimizations apply; it is up to the caller to ensure one doesn't
     violate the uniqueness of the keys (it'll happily crash in insert
     if you don't); when duplicate keys are allowed, a forward scan of
     the tree will visit them in the order of insertion.

   For a tree node:
     struct T {
       avlnode_t avlnode;
      char *key;
     };
   you could set the "indirect" flag, and then you simply use
   strcmp(), avoiding the need for passing templates in looking up key
   values. Much nicer.

   There is also an orthogonal variant that is enforced through the
   type system -- note that would be possible for all of the above as
   well, but the number of cases simply explodes and none of the above
   flags affects the dynamically changing data structures (just the
   tree definition), unlike this one.

   - the "C" variant keeps track of the number of nodes in the tree to
     support a "count" operation in O(1) time, but is otherwise
     identical.

   The various initializer macros and TreedefInit functions should
   make sense with this.

   All functions for looking up nodes return NULL if there is no node
   satisfying the requirements.

   - Init: initializes a tree (really just: root = NULL, perhaps count = 0)
   - Free: calls "freefun" on each node, which may free the node
   - FreeArg: as "Free", but with an extra, user-supplied, argument
   - Root: returns the root node
   - Lookup: returns a node with key value "key" (ref allowdups flag)
   - LookupIPath: like Lookup, but also filling an IPath_t structure
     for efficient insertion in case of a failed lookup (or inserting
     duplicates)
   - LookupDPath: like Lookup, but also filling a DPath_t structure
     that helps with deleting a node
   - LookupPredEq: locates the node with the greatest key value <= "key"
   - LookupSuccEq: similar, but smallest key value >= "key"
   - LookupPred: similar, < "key"
   - LookupSucc: similar, > "key"
   - Insert: convenience function: LookupIPath ; InsertIPath
   - Delete: convenience function: LookupDPath ; DeleteDPath
   - InsertIPath: insert node based on the "path" obtained from LookupIPath
   - DeleteDPath: delete node, using information in "path" to do so efficiently
   - SwapNode: replace "oldn" by "newn" without modifying the tree
     structure (the key need not be equal, but must be
     FindPred(oldn).key < newn.key < FindSucc(oldn).key, where a
     non-existing predecessor has key -inf and a non-existing
     successor has key +inf, and where it is understood that the <
     operator becomes <= if allowdups is set
   - AugmentUpdate: to be called when something in "node" changes that
     affects the subtree "summary" computed by the configured
     "augment" function
   - IsEmpty: returns 1 if tree is empty, 0 if not
   - IsSingleton: returns 1 if tree contains exactly one node, 0 if not
   - FindMin: returns the node with the smallest key value in the tree
   - FindMax: similar, largest key value
   - FindPred: preceding node in in-order treewalk
   - FindSucc: similar, following node

   - Walk: calls "f" with user-supplied argument "a" once for each
     node, starting at FindMin and ending at FindMax
   - ConstWalk: same, but with a const tree
   - WalkRange: like Walk, but only visiting nodes with key values in
     range [min,max] (that's inclusive)
   - ConstWalkRange: same, but with a const tree
   - WalkRangeReverse: like WalkRange, but in the reverse direction
   - ConstWalkRangeReverse: same, but with a const tree
   - IterFirst: starts forward iteration, starting at (and returning) FindMin
   - IterSuccEq: similar, starting at LookupSuccEq
   - IterSucc: similar, starting at LookupSucc
   - IterNext: returns FindSucc(last returned node); may not be called
     if preceding IterXXX call on same "iter" returned NULL

   That's all there is to it.

   Note that all calls to Walk(f,a) can be rewritten as:
     for(n=IterFirst(&it); n; n=IterNext(&it)) { f(n,a) }
   or as
     for(n=FindMin(); n; n=FindSucc(n)) { f(n,a) }

   The walk functions and iterators may not alter the tree
   structure. If that is desired, the latter can easily be rewritten
   as:
     n=FindMin() ; while(n) { nn=FindSucc(n); f(n,a); n=nn }
   because FindMin/FindSucc doesn't store any information to allow
   fast processing. That'll allow every operation, with the obvious
   exception of f(n) calling Delete(FindSucc(n)).

   Currently, all trees maintain parent pointers, but it may be worth
   doing a separate set without it, as it reduces the size of
   avlNode_t. But in that case, the FindMin/FindSucc option would no
   longer be a reasonable option because it would be prohibitively
   expensive, whereas the IterFirst/IterNext option are alway
   efficiently. If one were to do a threaded tree variant, the
   implemetantion of IterFirst/IterNext would become absolute trivial
   and faster still, but at the cost of significantly more overhead in
   memory and updates. */

#include <stdint.h>
#include <stdlib.h>

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSRT_AVL_MAX_TREEHEIGHT (12 * sizeof (void *))

typedef int (*ddsrt_avl_compare_t) (const void *a, const void *b);
typedef int (*ddsrt_avl_compare_r_t) (const void *a, const void *b, void *arg);
typedef void (*ddsrt_avl_augment_t) (void *node, const void *left, const void *right);
typedef void (*ddsrt_avl_walk_t) (void *node, void *arg);
typedef void (*ddsrt_avl_const_walk_t) (const void *node, void *arg);

typedef struct ddsrt_avl_node {
    struct ddsrt_avl_node *cs[2]; /* 0 = left, 1 = right */
    struct ddsrt_avl_node *parent;
    int height;
} ddsrt_avl_node_t;

#define DDSRT_AVL_TREEDEF_FLAG_INDKEY 1
#define DDSRT_AVL_TREEDEF_FLAG_R 2
#define DDSRT_AVL_TREEDEF_FLAG_ALLOWDUPS 4

typedef struct ddsrt_avl_treedef {
#if defined (__cplusplus)
    ddsrt_avl_treedef() {}
#endif
    size_t avlnodeoffset;
    size_t keyoffset;
    union {
        ddsrt_avl_compare_t comparekk;
        ddsrt_avl_compare_r_t comparekk_r;
    } u;
    ddsrt_avl_augment_t augment;
    uint32_t flags;
    void *cmp_arg; /* for _r variant */
} ddsrt_avl_treedef_t;

typedef struct ddsrt_avl_ctreedef {
    ddsrt_avl_treedef_t t;
} ddsrt_avl_ctreedef_t;

typedef struct ddsrt_avl_tree {
    ddsrt_avl_node_t *root;
} ddsrt_avl_tree_t;

typedef struct ddsrt_avl_ctree {
    ddsrt_avl_tree_t t;
    size_t count;
} ddsrt_avl_ctree_t;

typedef struct ddsrt_avl_path {
    int depth; /* total depth of path */
    int pnodeidx;
    ddsrt_avl_node_t *parent; /* (nodeidx == 0 ? NULL : *(path[nodeidx-1])) */
    ddsrt_avl_node_t **pnode[DDSRT_AVL_MAX_TREEHEIGHT];
} ddsrt_avl_path_t;

typedef struct ddsrt_avl_ipath {
    ddsrt_avl_path_t p;
} ddsrt_avl_ipath_t;

typedef struct ddsrt_avl_dpath {
    ddsrt_avl_path_t p;
} ddsrt_avl_dpath_t;

typedef struct ddsrt_avl_iter {
    const ddsrt_avl_treedef_t *td;
    ddsrt_avl_node_t *right;
    ddsrt_avl_node_t **todop;
    ddsrt_avl_node_t *todo[1+DDSRT_AVL_MAX_TREEHEIGHT];
} ddsrt_avl_iter_t;

typedef struct ddsrt_avl_citer {
    ddsrt_avl_iter_t t;
} ddsrt_avl_citer_t;

/* avlnodeoffset and keyoffset must both be in [0,2**31-1] */
#define DDSRT_AVL_TREEDEF_INITIALIZER(avlnodeoffset, keyoffset, comparekk_, augment) { (avlnodeoffset), (keyoffset), { .comparekk = (comparekk_) }, (augment), 0, 0 }
#define DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY(avlnodeoffset, keyoffset, comparekk_, augment) { (avlnodeoffset), (keyoffset), { .comparekk = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_INDKEY, 0 }
#define DDSRT_AVL_TREEDEF_INITIALIZER_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk_, augment) { (avlnodeoffset), (keyoffset), { .comparekk = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_ALLOWDUPS, 0 }
#define DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk_, augment) { (avlnodeoffset), (keyoffset), { .comparekk = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_INDKEY|DDSRT_AVL_TREEDEF_FLAG_ALLOWDUPS, 0 }
#define DDSRT_AVL_TREEDEF_INITIALIZER_R(avlnodeoffset, keyoffset, comparekk_, cmparg, augment) { (avlnodeoffset), (keyoffset), { .comparekk_r = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_R, (cmparg) }
#define DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_R(avlnodeoffset, keyoffset, comparekk_, cmparg, augment) { (avlnodeoffset), (keyoffset), { .comparekk_r = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_INDKEY|DDSRT_AVL_TREEDEF_FLAG_R, (cmparg) }
#define DDSRT_AVL_TREEDEF_INITIALIZER_R_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk_, cmparg, augment) { (avlnodeoffset), (keyoffset), { .comparekk_r = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_R|DDSRT_AVL_TREEDEF_FLAG_ALLOWDUPS, (cmparg) }
#define DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_R_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk_, cmparg, augment) { (avlnodeoffset), (keyoffset), { .comparekk_r = (comparekk_) }, (augment), DDSRT_AVL_TREEDEF_FLAG_INDKEY|DDSRT_AVL_TREEDEF_FLAG_R|DDSRT_AVL_TREEDEF_FLAG_ALLOWDUPS, (cmparg) }

/* Not maintaining # nodes */

DDS_EXPORT void ddsrt_avl_treedef_init (ddsrt_avl_treedef_t *td, size_t avlnodeoffset, size_t keyoffset, ddsrt_avl_compare_t comparekk, ddsrt_avl_augment_t augment, uint32_t flags) ddsrt_nonnull((1,4));
DDS_EXPORT void ddsrt_avl_treedef_init_r (ddsrt_avl_treedef_t *td, size_t avlnodeoffset, size_t keyoffset, ddsrt_avl_compare_r_t comparekk_r, void *cmp_arg, ddsrt_avl_augment_t augment, uint32_t flags) ddsrt_nonnull((1,4));

DDS_EXPORT void ddsrt_avl_init (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_free (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void (*freefun) (void *node)) ddsrt_nonnull((1,2));
DDS_EXPORT void ddsrt_avl_free_arg (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void (*freefun) (void *node, void *arg), void *arg) ddsrt_nonnull((1,2));

DDS_EXPORT void *ddsrt_avl_root (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_root_non_empty (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all ddsrt_attribute_returns_nonnull;
DDS_EXPORT void *ddsrt_avl_lookup (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_ipath (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key, ddsrt_avl_ipath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_dpath (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key, ddsrt_avl_dpath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_pred_eq (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_succ_eq (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_pred (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_lookup_succ (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *key) ddsrt_nonnull_all;

DDS_EXPORT void ddsrt_avl_insert (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void *node) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_delete (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void *node) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_insert_ipath (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void *node, ddsrt_avl_ipath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_delete_dpath (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void *node, ddsrt_avl_dpath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_swap_node (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, void *oldn, void *newn) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_augment_update (const ddsrt_avl_treedef_t *td, void *node) ddsrt_nonnull_all;

DDS_EXPORT int ddsrt_avl_is_empty (const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT int ddsrt_avl_is_singleton (const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;

DDS_EXPORT void *ddsrt_avl_find_min (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_find_max (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_find_pred (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *vnode) ddsrt_nonnull((1,2));
DDS_EXPORT void *ddsrt_avl_find_succ (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *vnode) ddsrt_nonnull((1,2));

DDS_EXPORT void ddsrt_avl_walk (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, ddsrt_avl_walk_t f, void *a) ddsrt_nonnull((1,2,3));
DDS_EXPORT void ddsrt_avl_const_walk (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3));
DDS_EXPORT void ddsrt_avl_walk_range (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, const void *min, const void *max, ddsrt_avl_walk_t f, void *a)  ddsrt_nonnull((1,2,3,4,5));
DDS_EXPORT void ddsrt_avl_const_walk_range (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *min, const void *max, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3,4,5));
DDS_EXPORT void ddsrt_avl_walk_range_reverse (const ddsrt_avl_treedef_t *td, ddsrt_avl_tree_t *tree, const void *min, const void *max, ddsrt_avl_walk_t f, void *a) ddsrt_nonnull((1,2,3));
DDS_EXPORT void ddsrt_avl_const_walk_range_reverse (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, const void *min, const void *max, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3));

DDS_EXPORT void *ddsrt_avl_iter_first (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, ddsrt_avl_iter_t *iter) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_iter_succ_eq (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, ddsrt_avl_iter_t *iter, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_iter_succ (const ddsrt_avl_treedef_t *td, const ddsrt_avl_tree_t *tree, ddsrt_avl_iter_t *iter, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_iter_next (ddsrt_avl_iter_t *iter) ddsrt_nonnull_all;

/* Maintaining # nodes */

#define DDSRT_AVL_CTREEDEF_INITIALIZER(avlnodeoffset, keyoffset, comparekk, augment) { DDSRT_AVL_TREEDEF_INITIALIZER (avlnodeoffset, keyoffset, comparekk, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_INDKEY(avlnodeoffset, keyoffset, comparekk, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY (avlnodeoffset, keyoffset, comparekk, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_ALLOWDUPS (avlnodeoffset, keyoffset, comparekk, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_INDKEY_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_ALLOWDUPS (avlnodeoffset, keyoffset, comparekk, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_R(avlnodeoffset, keyoffset, comparekk, cmparg, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_R (avlnodeoffset, keyoffset, comparekk, cmparg, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_INDKEY_R(avlnodeoffset, keyoffset, comparekk, cmparg, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_R (avlnodeoffset, keyoffset, comparekk, cmparg, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_R_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk, cmparg, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_R_ALLOWDUPS (avlnodeoffset, keyoffset, comparekk, cmparg, augment) }
#define DDSRT_AVL_CTREEDEF_INITIALIZER_INDKEY_R_ALLOWDUPS(avlnodeoffset, keyoffset, comparekk, cmparg, augment) { DDSRT_AVL_TREEDEF_INITIALIZER_INDKEY_R_ALLOWDUPS (avlnodeoffset, keyoffset, comparekk, cmparg, augment) }

DDS_EXPORT void ddsrt_avl_ctreedef_init (ddsrt_avl_ctreedef_t *td, size_t avlnodeoffset, size_t keyoffset, ddsrt_avl_compare_t comparekk, ddsrt_avl_augment_t augment, uint32_t flags) ddsrt_nonnull((1,4));
DDS_EXPORT void ddsrt_avl_ctreedef_init_r (ddsrt_avl_ctreedef_t *td, size_t avlnodeoffset, size_t keyoffset, ddsrt_avl_compare_r_t comparekk_r, void *cmp_arg, ddsrt_avl_augment_t augment, uint32_t flags) ddsrt_nonnull((1,4));

DDS_EXPORT void ddsrt_avl_cinit (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_cfree (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void (*freefun) (void *node)) ddsrt_nonnull((1,2));
DDS_EXPORT void ddsrt_avl_cfree_arg (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void (*freefun) (void *node, void *arg), void *arg) ddsrt_nonnull((1,2));

DDS_EXPORT void *ddsrt_avl_croot (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_croot_non_empty (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_ipath (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key, ddsrt_avl_ipath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_dpath (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key, ddsrt_avl_dpath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_pred_eq (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_succ_eq (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_pred (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_clookup_succ (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *key) ddsrt_nonnull_all;

DDS_EXPORT void ddsrt_avl_cinsert (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void *node) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_cdelete (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void *node) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_cinsert_ipath (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void *node, ddsrt_avl_ipath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_cdelete_dpath (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void *node, ddsrt_avl_dpath_t *path) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_cswap_node (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, void *oldn, void *newn) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_avl_caugment_update (const ddsrt_avl_ctreedef_t *td, void *node) ddsrt_nonnull_all;

DDS_EXPORT int ddsrt_avl_cis_empty (const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT int ddsrt_avl_cis_singleton (const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT size_t ddsrt_avl_ccount (const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;

DDS_EXPORT void *ddsrt_avl_cfind_min (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_cfind_max (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_cfind_pred (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *vnode) ddsrt_nonnull((1,2));
DDS_EXPORT void *ddsrt_avl_cfind_succ (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *vnode) ddsrt_nonnull((1,2));

DDS_EXPORT void ddsrt_avl_cwalk (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, ddsrt_avl_walk_t f, void *a) ddsrt_nonnull((1,2,3));
DDS_EXPORT void ddsrt_avl_cconst_walk (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3));
DDS_EXPORT void ddsrt_avl_cwalk_range (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, const void *min, const void *max, ddsrt_avl_walk_t f, void *a) ddsrt_nonnull((1,2,3,4,5));
DDS_EXPORT void ddsrt_avl_cconst_walk_range (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *min, const void *max, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3,4,5));
DDS_EXPORT void ddsrt_avl_cwalk_range_reverse (const ddsrt_avl_ctreedef_t *td, ddsrt_avl_ctree_t *tree, const void *min, const void *max, ddsrt_avl_walk_t f, void *a) ddsrt_nonnull((1,2,3,4,5));
DDS_EXPORT void ddsrt_avl_cconst_walk_range_reverse (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, const void *min, const void *max, ddsrt_avl_const_walk_t f, void *a) ddsrt_nonnull((1,2,3,4,5));

DDS_EXPORT void *ddsrt_avl_citer_first (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, ddsrt_avl_citer_t *iter) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_citer_succ_eq (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, ddsrt_avl_citer_t *iter, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_citer_succ (const ddsrt_avl_ctreedef_t *td, const ddsrt_avl_ctree_t *tree, ddsrt_avl_citer_t *iter, const void *key) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_avl_citer_next (ddsrt_avl_citer_t *iter) ddsrt_nonnull_all;

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_AVL_H */
