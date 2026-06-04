/*

MIT License

Copyright (c) 2024-2025 Vlad Krupinskii <mrvladus@yandex.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

------------------------------------------------------------------------------

xml.h - Header-only C/C++ library to parse XML.

------------------------------------------------------------------------------

Features:

- One tiny header file (~400 lines of code)
- Parses regular (`<tag>Tag Text</tag>`) and self-closing tags (`<tag/>`)
- Parses tags attributes (`<tag attribute="value" />`)
- Ignores comments and processing instructions

------------------------------------------------------------------------------

Usage:

Define XML_H_IMPLEMENTATION in ONE file before including "xml.h" to pull functions implementations.
In other files just include "xml.h".

#define XML_H_IMPLEMENTATION
#include "xml.h"

------------------------------------------------------------------------------

*/

#ifndef XML_H
#define XML_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct XMLNode XMLNode;
typedef struct XMLList XMLList;

// The main object to interact with parsed XML nodes. Represents single XML tag.
struct XMLNode {
  char *tag;         // Tag string.
  char *text;        // Inner text of the tag. NULL if tag has no inner text.
  XMLList *attrs;    // List of tag attributes. Check "node->attrs->len" if it has items.
  XMLNode *parent;   // Node's parent node. NULL for the root node.
  XMLList *children; // List of tag's sub-tags. Check "node->children->len" if it has items.
};

// Tags attribute containing key and value.
// Like: <tag foo_attr="foo_value" bar_attr="bar_value" />
typedef struct XMLAttr {
  char *key;
  char *value;
} XMLAttr;

// Simple dynamic list that only can grow in size.
// Used in XMLNode for list of children and list of tag attributes.
struct XMLList {
  size_t len;  // Length of the list.
  size_t size; // Capacity of the list in bytes.
  void **data; // List of pointers to list items.
};

// Parse XML string and return root XMLNode.
// Returns NULL for error.
XMLNode *xml_parse_string(const char *xml);
// Parse XML file for given path and return root XMLNode.
// Returns NULL for error.
XMLNode *xml_parse_file(const char *path);
// Get child of the node at index.
// Returns NULL if not found.
XMLNode *xml_node_child_at(XMLNode *node, size_t idx);
// Find xml tag by path like "div/p/href"
XMLNode *xml_node_find_by_path(XMLNode *root, const char *path, bool exact);
// Get first matching tag in the tree.
// If exact is "false" - will return first tag which name contains "tag"
XMLNode *xml_node_find_tag(XMLNode *node, const char *tag, bool exact);
// Get value of the tag attribute.
// Returns NULL if not found.
const char *xml_node_attr(XMLNode *node, const char *attr_key);
// Cleanup node and all it's children.
void xml_node_free(XMLNode *node);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // XML_H

#ifdef XML_H_IMPLEMENTATION

#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------ UTILS ------ //

// Print debug message.
// Only if XML_H_DEBUG is defined.
void log_debug(const char *format, ...) {
  (void)format; // Avoid unused parameter warning
#ifdef XML_H_DEBUG
  va_list args;
  printf("[XML.H DEBUG] ");
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  printf("\n");
#endif
}

void log_error(const char *format, ...) {
  va_list args;
  printf("[XML.H ERROR] ");
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  printf("\n");
}

// ------ LIST ------ //

// Create new dynamic array
XMLList *xml_list_new() {
  XMLList *list = (XMLList *)malloc(sizeof(XMLList));
  list->len = 0;
  list->size = 1;
  list->data = (void **)malloc(sizeof(void *) * list->size);
  return list;
}

// Add element to the end of the array. Grow if needed.
void xml_list_add(XMLList *list, void *data) {
  if (list->len >= list->size) {
    list->size *= 2; // Exponential growth
    list->data = (void **)realloc(list->data, list->size * sizeof(void *));
  }
  list->data[list->len++] = data;
}

// ------ NODE ------ //

// Create new XMLNode. Free with xml_node_free().
static XMLNode *xml_node_new(XMLNode *parent) {
  XMLNode *node = (XMLNode *)malloc(sizeof(XMLNode));
  node->parent = parent;
  node->tag = NULL;
  node->text = NULL;
  node->children = xml_list_new();
  node->attrs = xml_list_new();
  if (parent) xml_list_add(parent->children, node);
  return node;
}

// Add XMLAttr to the node's list of attributes
static void xml_node_add_attr(XMLNode *node, char *key, char *value) {
  XMLAttr *attr = (XMLAttr *)malloc(sizeof(XMLAttr));
  attr->key = key;
  attr->value = value;
  xml_list_add(node->attrs, attr);
}

XMLNode *xml_node_child_at(XMLNode *node, size_t index) {
  if (index > node->children->len - 1) return NULL;
  return (XMLNode *)node->children->data[index];
}

XMLNode *xml_node_find_tag(XMLNode *node, const char *tag, bool exact) {
  if (!node || !tag) return NULL; // Invalid input
  // Check if the current node matches the tag
  if (node->tag && ((exact && strcmp(node->tag, tag) == 0) || (!exact && strstr(node->tag, tag) != NULL))) return node;
  // Recursively search through the children of the node
  for (size_t i = 0; i < node->children->len; i++) {
    XMLNode *result = xml_node_find_tag((XMLNode *)node->children->data[i], tag, exact);
    if (result) return result; // Return the first match found
  }
  // No match found in this subtree
  return NULL;
}

XMLNode *xml_node_find_by_path(XMLNode *root, const char *path, bool exact) {
  if (!root || !path) return NULL;
  char *tokenized_path = strdup(path);
  if (!tokenized_path) return NULL;
  char *segment = strtok(tokenized_path, "/");
  XMLNode *current = root;
  while (segment && current) {
    bool found = false;
    for (size_t i = 0; i < current->children->len; i++) {
      XMLNode *child = (XMLNode *)current->children->data[i];
      if ((exact && strcmp(child->tag, segment) == 0) || (!exact && strstr(child->tag, segment) != NULL)) {
        current = child;
        found = true;
        break;
      }
    }
    if (!found) {
      free(tokenized_path);
      return NULL;
    }
    segment = strtok(NULL, "/");
  }
  free(tokenized_path);
  return current;
}

const char *xml_node_attr(XMLNode *node, const char *attr_key) {
  if (!node || !attr_key) return NULL;
  for (size_t i = 0; i < node->attrs->len; i++) {
    XMLAttr *attr = (XMLAttr *)node->attrs->data[i];
    if (!strcmp(attr->key, attr_key)) return attr->value;
  }
  return NULL;
}

void xml_node_free(XMLNode *node) {
  if (node == NULL) return;
  // Free text
  if (node->text) free(node->text);
  // Free the attributes
  for (size_t i = 0; i < node->attrs->len; i++) {
    XMLAttr *attr = (XMLAttr *)node->attrs->data[i];
    free(attr->key);
    free(attr->value);
    free(attr);
  }
  free(node->attrs->data);
  free(node->attrs);
  // Recursively free the children
  for (size_t i = 0; i < node->children->len; i++) xml_node_free((XMLNode *)node->children->data[i]);
  free(node->children->data);
  free(node->children);
  // Free the tag
  if (node->tag) free(node->tag);
  // Free the node itself
  free(node);
}

// ------ PARSING FUNCTIONS ------ //

// Trim leading and trailing whitespace from a string (in place).
static void trim_text(char *text) {
  char *start = text;
  while (*start && isspace((unsigned char)*start)) start++;
  char *dest = text;
  while (*start) *dest++ = *start++;
  *dest = '\0';
  dest = text + strlen(text) - 1;
  while (dest >= text && isspace((unsigned char)*dest)) {
    *dest = '\0';
    dest--;
  }
}

static void skip_whitespace(const char *xml, size_t *idx) {
  while (isspace(xml[*idx])) (*idx)++;
}

// Skip processing instructions and comments
// Returns true if skipped.
static bool skip_tags(const char *xml, size_t *idx) {
  size_t start = *idx;
  // Processing instruction
  if (xml[*idx] == '?') {
    while (!(xml[*idx] == '>' && xml[*idx - 1] == '?')) (*idx)++;
    (*idx)++;
    log_debug("Parsed processing instruction: <%.*s", (int)(*idx - start), xml + start);
    return true;
  }
  // Comment
  else if (xml[*idx] == '!' && xml[*idx + 1] == '-' && xml[*idx + 2] == '-') {
    while (!(xml[*idx] == '>' && xml[*idx - 1] == '-' && xml[*idx - 2] == '-')) (*idx)++;
    (*idx)++;
    log_debug("Parsed comment: <%.*s", (int)(*idx - start), xml + start);
    return true;
  }
  return false;
}

// Parse end tag </tag>.
static void parse_end_tag(const char *xml, size_t *idx, XMLNode **curr_node) {
  (*idx)++; // Skip '/'
  skip_whitespace(xml, idx);
  while (xml[*idx] != '>') (*idx)++;
  (*idx)++; // Skip '>'
  log_debug("Parsed end tag: </%s>", (*curr_node)->tag);
  *curr_node = (*curr_node)->parent;
}

// Parse tag name <name ... >
static void parse_tag_name(const char *xml, size_t *idx, XMLNode **curr_node) {
  size_t tag_start = *idx;
  while (!(isspace(xml[*idx]) || xml[*idx] == '>' || xml[*idx] == '/')) (*idx)++;
  (*curr_node)->tag = strndup(xml + tag_start, *idx - tag_start);
}

// Parse tag attributes <tag attr="value" ... >
static void parse_tag_attributes(const char *xml, size_t *idx, XMLNode **curr_node) {
  skip_whitespace(xml, idx);
  while (!(xml[*idx] == '>' || xml[*idx] == '/')) {
    // Get attribute key
    size_t attr_start = *idx;
    while (!(xml[*idx] == '=' || isspace(xml[*idx]))) (*idx)++;
    char *attr_key = strndup(xml + attr_start, *idx - attr_start);
    // Skip ="
    if (isspace(xml[*idx])) skip_whitespace(xml, idx);
    else if (xml[*idx] == '=') {
      (*idx)++;
      skip_whitespace(xml, idx);
    }
    char quote = xml[*idx];
    (*idx)++; // Skip opening quote
    size_t value_start = *idx;
    while (!(xml[*idx] == quote && xml[*idx - 1] != '\\')) (*idx)++;
    char *attr_value = strndup(xml + value_start, *idx - value_start);
    (*idx)++; // Skip closing quote
    xml_node_add_attr(*curr_node, attr_key, attr_value);
    skip_whitespace(xml, idx);
    log_debug("Parsed tag attribute: <%s %s=\"%s\">", (*curr_node)->tag, attr_key, attr_value);
  }
}

static void parse_tag_inner_text(const char *xml, size_t *idx, XMLNode **curr_node) {
  size_t text_start = *idx;
  while (xml[*idx] != '<') (*idx)++;
  if (*idx > text_start) {
    (*curr_node)->text = strndup(xml + text_start, *idx - text_start);
    trim_text((*curr_node)->text);
    log_debug("Parsed inner text for <%s>: %s", (*curr_node)->tag, (*curr_node)->text);
  }
}

// Parse start tag.
// Returns false if the tag is self-closing like: <tag />
// Call continue if returns false.
static bool parse_tag(const char *xml, size_t *idx, XMLNode **curr_node) {
  skip_whitespace(xml, idx);
  // End tag </tag>
  if (xml[*idx] == '/') {
    parse_end_tag(xml, idx, curr_node);
    return false;
  }
  // Create new node with current curr_node as parent
  *curr_node = xml_node_new(*curr_node);
  // Start tag <tag...>
  parse_tag_name(xml, idx, curr_node);
  // Parse attributes
  if (isspace(xml[*idx])) parse_tag_attributes(xml, idx, curr_node);
  // Self-closing tag <tag ... />
  if (xml[*idx] == '/') {
    (*idx)++; // Skip '/'
    while (xml[*idx] != '>') (*idx)++;
    (*idx)++; // Consume '>'
    log_debug("Parsed self-closing tag: <%s/>", (*curr_node)->tag);
    *curr_node = (*curr_node)->parent;
    return false;
  }
  // Start tag <tag ... >
  else if (xml[*idx] == '>') {
    (*idx)++; // Consume '>'
    log_debug("Parsed start tag: <%s>", (*curr_node)->tag);
    skip_whitespace(xml, idx);
    parse_tag_inner_text(xml, idx, curr_node);
    // If the next character is '<', parse the next tag
    if (xml[*idx] == '<') {
      (*idx)++; // Consume '<'
      return parse_tag(xml, idx, curr_node);
    }
    return true;
  }
  skip_whitespace(xml, idx);
  return true;
}

XMLNode *xml_parse_string(const char *xml) {
  log_debug("Parse string: %s", xml);
  XMLNode *root = xml_node_new(NULL);
  XMLNode *curr_node = root;
  size_t idx = 0;
  while (xml[idx] != '\0') {
    skip_whitespace(xml, &idx);
    // Parse tag
    if (xml[idx] == '<') {
      idx++;
      skip_whitespace(xml, &idx);
      if (skip_tags(xml, &idx)) continue;
      if (!parse_tag(xml, &idx, &curr_node)) continue;
    }
    idx++;
  }
  log_debug("Done parsing string");
  return root;
}

XMLNode *xml_parse_file(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) return NULL;
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *buffer = (char *)malloc(file_size + 1);
  if (!buffer) {
    fclose(file);
    return NULL;
  }
  size_t bytes_read = fread(buffer, 1, file_size, file);
  if (bytes_read != (size_t)file_size) {
    free(buffer);
    fclose(file);
    return NULL;
  }
  buffer[file_size] = '\0';
  fclose(file);
  XMLNode *node = xml_parse_string(buffer);
  free(buffer);
  return node;
}

#endif // XML_H_IMPLEMENTATION

#ifdef XML_H_TEST

// To run tests compile with:
// $ cp xml.h xml.c && cc -DXML_H_IMPLEMENTATION -DXML_H_TEST -DXML_H_DEBUG -o test xml.c && rm xml.c
// $ ./test

const char test_xml[] = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>"
                        "<!-- Test comment -->"
                        "<library>"
                        "    <book id=\"1\">"
                        "        <title>The Great Gatsby</title>"
                        "        <author>F. Scott Fitzgerald</author>"
                        "        <rating value=\"4.5\" />"
                        "    </book>"
                        "    <book id=\"2\">"
                        "        <title>1984</title>"
                        "        <author>George Orwell</author>"
                        "        <rating value=\"4.9\" />"
                        "        <bestseller />"
                        "    </book>"
                        "</library>";

int main() {
  printf("Test XML:\n%s\n", test_xml);
  // Parse XML string and get root node
  XMLNode *root = xml_parse_string(test_xml);
  // Get first child of the root node
  XMLNode *library = xml_node_child_at(root, 0);
  // Print ratings of all books
  for (size_t i = 0; i < library->children->len; i++) {
    // Get book
    XMLNode *book = xml_node_child_at(library, i);
    for (size_t j = 0; j < book->children->len; j++) {
      // Get book child tag
      XMLNode *sub_tag = xml_node_child_at(book, j);
      // If tag name is "rating" - print its value
      if (!strcmp(sub_tag->tag, "rating")) {
        // Get rating value
        const char *rating = xml_node_attr(sub_tag, "value");
        printf("Rating is %s\n", rating);
      }
    }
  }
  // Find 1st matching tag
  XMLNode *title_tag = xml_node_find_tag(root, "title", true);
  printf("Matching tag for 'title' = '%s'\n", title_tag->tag);
  // Find 1st matching tag by path
  XMLNode *author_tag = xml_node_find_by_path(root, "library/book/author", true);
  printf("Matching tag for 'author' = '%s'\n", author_tag->tag);
  // Cleanup and exit
  xml_node_free(root);
  return 0;
}

#endif // XML_H_TEST
