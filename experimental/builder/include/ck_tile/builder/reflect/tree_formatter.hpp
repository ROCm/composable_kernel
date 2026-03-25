// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <deque>
#include <sstream>
#include <string>

namespace ck_tile::reflect {

// Tree-node class for building hierarchical tree structures, then rendering them
// with proper indentation and tree-drawing characters (├─, └─, │, etc.)
//
// Unlike a streaming API, the tree is built first and rendered afterwards,
// so last-child status is determined automatically.
//
// Example Usage:
//
//   TreeFormatter root("Root");
//   auto& b1 = root.add("Branch 1");
//   b1.add("Item 1a");
//   b1.add("Item 1b");
//   auto& b2 = root.add("Branch 2");
//   b2.add("Item 2a");
//   std::cout << root.getString() << "\n";
//
// Generated Output:
//
//   Root
//   ├─ Branch 1
//   │  ├─ Item 1a
//   │  └─ Item 1b
//   └─ Branch 2
//      └─ Item 2a
class TreeFormatter
{
    public:
    // Construct a node with content built from the given arguments
    template <typename... Args>
    explicit TreeFormatter(Args&&... args)
    {
        std::ostringstream oss;
        ((oss << std::forward<Args>(args)), ...);
        content_ = oss.str();
    }

    // Add a child node, returns a reference to it for further nesting
    template <typename... Args>
    TreeFormatter& add(Args&&... args) [[clang::lifetimebound]]
    {
        children_.emplace_back(std::forward<Args>(args)...);
        return children_.back();
    }

    // Render the full tree to a string
    std::string getString() const
    {
        std::ostringstream oss;
        oss << content_;
        for(size_t i = 0; i < children_.size(); ++i)
        {
            oss << '\n';
            children_[i].renderChild(oss, "", i == children_.size() - 1);
        }
        return oss.str();
    }

    private:
    std::string content_;
    // std::deque preserves references to existing elements on push_back/emplace_back,
    // unlike std::vector which may reallocate. This allows add() to safely return
    // a reference to the newly added child for further nesting.
    std::deque<TreeFormatter> children_;

    // Recursive render helper
    void renderChild(std::ostringstream& oss, const std::string& prefix, bool is_last) const
    {
        oss << prefix << (is_last ? "└─ " : "├─ ") << content_;
        std::string child_prefix = prefix + (is_last ? "   " : "│  ");
        for(size_t i = 0; i < children_.size(); ++i)
        {
            oss << '\n';
            children_[i].renderChild(oss, child_prefix, i == children_.size() - 1);
        }
    }
};

} // namespace ck_tile::reflect
