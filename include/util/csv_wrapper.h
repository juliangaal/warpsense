#pragma once

#include <vector>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>

template <typename T>
class CSVWrapper
{
public:
    using CSVRow = std::vector<T>;
    using Header = std::vector<std::string>;

    template <typename S>
    struct _CSVObject
    {
        std::string name;
        Header header;
        std::vector<CSVRow> rows;

        explicit _CSVObject(const std::string& name)
        {
            this->name = name;
        }

        inline void add_row(const CSVRow& row)
        {
            this->rows.push_back(row);
        }

        inline void set_header(const Header& header_data)
        {
            this->header = header_data;
        }
    };

    using CSVObject = _CSVObject<T>;

    explicit CSVWrapper(const std::filesystem::path& dir, const std::string descript = "", char sep_char = ',')
    : save_dir_(dir)
    , descriptor_(descript)
    , seperator_(sep_char)
    {
    }
    
    ~CSVWrapper()
    {
      write_all();
    }

    inline CSVObject* get_instance(const std::string& name)
    {
        auto obj_it = csv_objects.find(name);
        if(obj_it != csv_objects.end())
        {
            return &obj_it->second;
        }

        csv_objects.emplace(name, CSVObject(name));
        return &csv_objects.at(name);
    }

    template <typename S>
    inline void write_row(const std::vector<S>& row, std::ofstream& file)
    {
        for (size_t i = 0; i < row.size(); i++)
        {
          const auto& val = row[i];
          file << val;
          if(i < row.size() - 1)
          {
            file << seperator_;
          }
        }
        file << "\n";
    }

    /**
     * @brief writes all csv objects to the specified location
     * 
     */
    inline void write_all()
    {
      std::time_t t = std::time(nullptr);
      std::tm tm = *std::localtime(&t);

      std::stringstream ss;
      ss << std::put_time(&tm, "%H-%M-%S_%d-%m-%Y");

      std::filesystem::path base_path;

      if (descriptor_.empty())
      {
        base_path = save_dir_ / std::filesystem::path(ss.str());
      }
      else
      {
        base_path = save_dir_ / std::filesystem::path(descriptor_ + "_" + ss.str());
      }
      std::filesystem::create_directory(base_path);
      std::cout << "Creating CSV in directory: " << base_path << std::endl;

      for(const auto& [name, object]: csv_objects)
      {
        std::filesystem::path file_path(name);
        file_path.replace_extension("csv");
        file_path = base_path / file_path;

        std::ofstream file;

        file.open(file_path, std::ios::out | std::ios::trunc);

        if(!file.is_open())
        {
          std::cout << "Could not create file with path: " << file_path << std::endl;
          continue;
        }

        write_row(object.header, file);
        for (const auto &row: object.rows)
        {
          write_row(row, file);
        }

        file.close();
      }
    }

private:
    
    std::unordered_map<std::string, CSVObject> csv_objects;
    //std::vector<CSVObject> csv_objects;
    std::filesystem::path save_dir_;
    std::string descriptor_;
    char seperator_;
};

