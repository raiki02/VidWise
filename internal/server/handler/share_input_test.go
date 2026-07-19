package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestNormalizeVideoShareInputParsesSupportedShareText(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantURL   string
		wantTitle string
	}{
		{
			name:      "bilibili",
			input:     `【为什么说有了男女之事以后就不会再长痘痘了？】 https://www.bilibili.com/video/BV1wHK36JEPT/?share_source=copy_web&vd_source=12f149fc8976b8efb329aa283d7b827b`,
			wantURL:   "https://www.bilibili.com/video/BV1wHK36JEPT/?share_source=copy_web&vd_source=12f149fc8976b8efb329aa283d7b827b",
			wantTitle: "为什么说有了男女之事以后就不会再长痘痘了？",
		},
		{
			name:      "douyin",
			input:     `0.05 11/01 aAG:/ g@B.te :2pm 别被毒鸡汤毁了... # chiikawa # 毒鸡汤 # 吉伊卡哇 # 青年创作者成长计划  https://v.douyin.com/MCJ9MnV0_jY/ 复制此链接，打开Dou音搜索，直接观看视频！`,
			wantURL:   "https://v.douyin.com/MCJ9MnV0_jY/",
			wantTitle: "别被毒鸡汤毁了...",
		},
		{
			name:      "xiaohongshu",
			input:     `54 【满身蜱虫痒到发疯！非洲水牛下河，万条小鱼集体上门搓澡 - 狂野星球说 | 小红书 - 你的生活兴趣社区】 😆 R2mLHAQeGI3Satn 😆 https://www.xiaohongshu.com/discovery/item/6a4cc74500000000150279a4?source=webshare&xhsshare=pc_web&xsec_token=ABYkvtp8_gX1Tai24ZgtvB7r9gvNP1yaSHYWH10TE-Zv0=&xsec_source=pc_share,`,
			wantURL:   "https://www.xiaohongshu.com/discovery/item/6a4cc74500000000150279a4?source=webshare&xhsshare=pc_web&xsec_token=ABYkvtp8_gX1Tai24ZgtvB7r9gvNP1yaSHYWH10TE-Zv0=&xsec_source=pc_share",
			wantTitle: "满身蜱虫痒到发疯！非洲水牛下河，万条小鱼集体上门搓澡 - 狂野星球说",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeVideoShareInput(tt.input, "")
			if got.URL != tt.wantURL {
				t.Fatalf("URL = %q, want %q", got.URL, tt.wantURL)
			}
			if got.Title != tt.wantTitle {
				t.Fatalf("Title = %q, want %q", got.Title, tt.wantTitle)
			}
			if got.Name != sanitizeName(tt.wantTitle) {
				t.Fatalf("Name = %q, want sanitized title %q", got.Name, sanitizeName(tt.wantTitle))
			}
		})
	}
}

func TestNormalizeVideoShareInputPrefersExplicitName(t *testing.T) {
	got := normalizeVideoShareInput(
		`【男男的一生】 https://www.bilibili.com/video/BV1oBNR6YEo9/?share_source=copy_web&vd_source=12f149fc8976b8efb329aa283d7b827b`,
		" my demo? ",
	)

	if got.URL != "https://www.bilibili.com/video/BV1oBNR6YEo9/?share_source=copy_web&vd_source=12f149fc8976b8efb329aa283d7b827b" {
		t.Fatalf("URL = %q", got.URL)
	}
	if got.Title != "男男的一生" {
		t.Fatalf("Title = %q, want parsed share title", got.Title)
	}
	if got.Name != "my_demo" {
		t.Fatalf("Name = %q, want explicit sanitized name", got.Name)
	}
}

func TestBindExtractRequestParsesShareTextName(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body, err := json.Marshal(map[string]string{
		"url":  `7.97 :2pm 12/24 T@l.pD Mjc:/ 权贵用钱篡改司法，全民线上死刑投票 # 影视解说 # 惊悚 # 悬疑 # 国民死刑投票 # 抖音精选  https://v.douyin.com/TLzeo_ZpixQ/ 复制此链接，打开Dou音搜索，直接观看视频！`,
		"type": "audio",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/extract", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	got, err := bindExtractRequest(c)
	if err != nil {
		t.Fatalf("bindExtractRequest: %v", err)
	}
	if got.URL != "https://v.douyin.com/TLzeo_ZpixQ/" {
		t.Fatalf("URL = %q", got.URL)
	}
	if got.Name != "权贵用钱篡改司法_全民线上死刑投票" {
		t.Fatalf("Name = %q", got.Name)
	}
	if got.Type != "audio" {
		t.Fatalf("Type = %q", got.Type)
	}
}
