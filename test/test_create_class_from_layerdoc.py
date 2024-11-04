#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
try:
    import marko
except ImportError:
    marko = None


def do_heading(parse, field_str_open) -> bool:
    if field_str_open:
        print('        """,')
        print("    )")
        print()
        field_str_open = False
    if parse.level == 5:
        codespan = parse.children[0]
        if len(parse.children) == 2:
            importance = parse.children[1]
        else:
            importance = marko.inline.RawText("")
        assert isinstance(codespan, marko.inline.CodeSpan)
        assert isinstance(importance, marko.inline.RawText)
        fieldtype = "int"
        is_optional = (
            "\n        optional=True,"
            if "optional" in importance.children.lower()
            else ""
        )
        field_str = f"""    {codespan.children}: {fieldtype} = Field(
    default=0,{is_optional}
    description=\"\"\""""
        field_str_open = True
        print(field_str, end="")
    return field_str_open


def do_paragraph(parse, field_str_open) -> bool:
    if isinstance(parse.children, str):
        print(parse.children)
    elif isinstance(parse.children, list):
        for child in parse.children:
            if isinstance(child, marko.inline.CodeSpan):
                print(f"`{child.children}`", end="")
            elif isinstance(child, marko.inline.RawText):
                print(child.children, end=" ")
            elif isinstance(child, marko.inline.Emphasis):
                print("* ... *", end="")
            elif isinstance(child, marko.inline.Link):
                continue
            elif isinstance(child, marko.inline.LineBreak):
                print()
            else:
                ...


def main():
    # Read the README.md file
    with open("netdeployonnx/devices/max78000/ai8xize/layerdoc.md") as file:
        markdown_content = file.read()
    # Parse the markdown content
    parsed = marko.parse(markdown_content)
    field_str_open = False
    for parse in parsed.children:
        if isinstance(parse, marko.block.BlankLine):
            continue  # ignore blank lines
        elif isinstance(parse, marko.block.FencedCode):
            continue  # ignore code blocks
        elif isinstance(parse, marko.inline.LineBreak):
            continue  # ignore newlines
        elif isinstance(parse, marko.block.Heading):
            field_str_open = do_heading(parse, field_str_open)
        elif isinstance(parse, marko.block.Paragraph):
            field_str_open = do_paragraph(parse, field_str_open)
        elif isinstance(parse, marko.block.List):
            ...
        else:
            print(type(parse))
    if field_str_open:
        print('        """,')
        print("    )")
        print()
        field_str_open = False


if __name__ == "__main__":
    main()
