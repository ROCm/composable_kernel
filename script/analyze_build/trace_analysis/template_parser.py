# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Template signature parser for C++ templates.

Parses template signatures to extract structure, arguments, and relationships.
"""

from typing import List, Tuple
import re


class TemplateParser:
    """
    Parser for C++ template signatures.

    Extracts template structure including base name, arguments, and nesting depth.
    Handles complex nested templates with proper bracket matching.
    """

    @staticmethod
    def parse_signature(signature: str) -> Tuple[str, List[str], int]:
        """
        Parse a template signature into components.

        Args:
            signature: Full template signature string

        Returns:
            Tuple of (base_name, arguments, depth)
            - base_name: Template name before '<'
            - arguments: List of argument strings
            - depth: Maximum nesting depth

        Examples:
            >>> parse_signature("std::vector<int>")
            ("std::vector", ["int"], 1)

            >>> parse_signature("MyClass<T, std::vector<int>>")
            ("MyClass", ["T", "std::vector<int>"], 2)
        """
        signature = signature.strip()

        # Check if this is a template (contains '<')
        if "<" not in signature:
            return signature, [], 0

        # Find the base name (everything before first '<')
        first_bracket = signature.index("<")
        base_name = signature[:first_bracket].strip()

        # Extract arguments between outermost '<' and '>'
        args = TemplateParser._extract_arguments(signature[first_bracket:])

        # Compute maximum nesting depth
        depth = TemplateParser._compute_depth(signature)

        return base_name, args, depth

    @staticmethod
    def _extract_arguments(template_part: str) -> List[str]:
        """
        Extract template arguments from the template part (starting with '<').

        Handles nested templates and comma-separated arguments correctly.

        Args:
            template_part: String starting with '<' containing template arguments

        Returns:
            List of argument strings
        """
        if not template_part.startswith("<"):
            return []

        # Find matching closing bracket
        depth = 0
        end_pos = -1
        for i, char in enumerate(template_part):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos == -1:
            # Malformed template, return empty
            return []

        # Extract content between outermost brackets
        content = template_part[1:end_pos].strip()

        if not content:
            return []

        # Split by commas, respecting nested brackets
        args = []
        current_arg = []
        depth = 0

        for char in content:
            if char == "<":
                depth += 1
                current_arg.append(char)
            elif char == ">":
                depth -= 1
                current_arg.append(char)
            elif char == "," and depth == 0:
                # Top-level comma, split here
                args.append("".join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)

        # Add the last argument
        if current_arg:
            args.append("".join(current_arg).strip())

        return args

    @staticmethod
    def _compute_depth(signature: str) -> int:
        """
        Compute the maximum nesting depth of template brackets.

        Args:
            signature: Template signature string

        Returns:
            Maximum nesting depth (0 if no templates)
        """
        max_depth = 0
        current_depth = 0

        for char in signature:
            if char == "<":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ">":
                current_depth -= 1

        return max_depth

    @staticmethod
    def is_template_argument(arg: str) -> bool:
        """
        Check if an argument is itself a template.

        Args:
            arg: Argument string

        Returns:
            True if argument contains template brackets
        """
        return "<" in arg and ">" in arg

    @staticmethod
    def classify_argument(arg: str) -> str:
        """
        Classify a template argument type.

        Args:
            arg: Argument string

        Returns:
            'template', 'primitive', or 'unknown'
        """
        arg = arg.strip()

        if TemplateParser.is_template_argument(arg):
            return "template"

        # Common primitive types
        primitives = {
            "int",
            "long",
            "short",
            "char",
            "float",
            "double",
            "bool",
            "void",
            "size_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
        }

        # Remove const, volatile, unsigned, signed
        cleaned = re.sub(r"\b(const|volatile|unsigned|signed)\b", "", arg).strip()

        # Check if it's a primitive
        if cleaned in primitives:
            return "primitive"

        # Check for numeric literals
        if arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
            return "primitive"

        return "unknown"

    @staticmethod
    def extract_template_hierarchy(signature: str) -> List[str]:
        """
        Extract all template signatures in a nested template.

        Returns a list of all template signatures found, from outermost to innermost.

        Args:
            signature: Full template signature

        Returns:
            List of template signatures

        Example:
            >>> extract_template_hierarchy("A<B<C>, D<E>>")
            ["A<B<C>, D<E>>", "B<C>", "D<E>", "C", "E"]
        """
        templates = []

        # Add the full signature if it's a template
        if "<" in signature:
            templates.append(signature)

        # Extract arguments and recurse
        _, args, _ = TemplateParser.parse_signature(signature)

        for arg in args:
            if TemplateParser.is_template_argument(arg):
                # Recursively extract from nested templates
                nested = TemplateParser.extract_template_hierarchy(arg)
                templates.extend(nested)

        return templates

    @staticmethod
    def normalize_signature(signature: str) -> str:
        """
        Normalize a template signature for comparison.

        Removes extra whitespace and standardizes formatting.

        Args:
            signature: Template signature

        Returns:
            Normalized signature
        """
        # Remove extra whitespace
        signature = re.sub(r"\s+", " ", signature)

        # Remove spaces around < and >
        signature = re.sub(r"\s*<\s*", "<", signature)
        signature = re.sub(r"\s*>\s*", ">", signature)

        # Remove spaces around commas
        signature = re.sub(r"\s*,\s*", ",", signature)

        return signature.strip()
