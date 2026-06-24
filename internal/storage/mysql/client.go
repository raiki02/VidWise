package mysql

import (
	"fmt"
	"log/slog"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

type Client struct {
	DB *gorm.DB
}

func NewClient(dsn string, maxOpen, maxIdle int) (*Client, error) {
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{
		Logger:                                   logger.Default.LogMode(logger.Warn),
		SkipDefaultTransaction:                   false,
		DisableForeignKeyConstraintWhenMigrating: true,
	})
	if err != nil {
		return nil, fmt.Errorf("open mysql: %w", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("get underlying sql.DB: %w", err)
	}
	sqlDB.SetMaxOpenConns(maxOpen)
	sqlDB.SetMaxIdleConns(maxIdle)

	slog.Info("mysql.connected")
	return &Client{DB: db}, nil
}

// AutoMigrate runs GORM auto-migration for all domain models.
func (c *Client) AutoMigrate(models ...any) error {
	if err := c.DB.AutoMigrate(models...); err != nil {
		return fmt.Errorf("auto-migrate: %w", err)
	}
	slog.Info("mysql.migrated")
	return nil
}

func (c *Client) Close() error {
	sqlDB, err := c.DB.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
