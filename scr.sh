#!/bin/bash
        find ./ -name "*.txt" -type f | \
        (while read file;
            do if [[ "$file" != *.DS_Store* ]]; then
            if [[ "$file" != *-utf8* ]]; then
                iconv -f ISO-8859-1 -t UTF-8 "$file" > "$file-utf8";
                rm $file;
                echo mv "$file-utf8" "$file";
                mv "$file-utf8" "$file";
            fi
        fi 
        done);